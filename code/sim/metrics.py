import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score as sk_silhouette
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def kl_gaussian(mean1, cov1, mean0, cov0):
    """KL[ N(mean1,cov1) || N(mean0,cov0) ]."""
    mean1, cov1 = np.atleast_1d(mean1), np.atleast_2d(cov1)
    mean0, cov0 = np.atleast_1d(mean0), np.atleast_2d(cov0)
    d = mean1.shape[0]
    try:
        cov0_inv = np.linalg.pinv(cov0)
        s0, ld0  = np.linalg.slogdet(cov0)
        s1, ld1  = np.linalg.slogdet(cov1)
        if s0 <= 0 or s1 <= 0:
            return 0.0
        diff = mean0 - mean1
        kl   = 0.5 * (np.trace(cov0_inv @ cov1) +
                      diff @ cov0_inv @ diff - d + ld0 - ld1)
        return max(0.0, float(kl))
    except np.linalg.LinAlgError:
        return 0.0


def ndcg_at_k(recommendations: dict, preference_matrix: np.ndarray, K: int) -> float:
    """NDCG@K по матрице предпочтений."""
    n_users, n_items = preference_matrix.shape
    scores = []
    for uid, items in recommendations.items():
        if uid >= n_users:
            continue
        true_rel = preference_matrix[uid]
        items_k  = [i for i in items[:K] if i < n_items]
        dcg   = sum(true_rel[i] / np.log2(r + 2) for r, i in enumerate(items_k))
        ideal = sorted(true_rel, reverse=True)[:K]
        idcg  = sum(v / np.log2(r + 2) for r, v in enumerate(ideal))
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def mean_intra_cluster_jaccard(recommendations: dict,
                                cluster_labels: np.ndarray,
                                n_pairs: int = 200) -> float:
    """Среднее Jaccard-сходство рекомендаций внутри одного кластера."""
    unique_ks = np.unique(cluster_labels)
    scores = []
    rng = np.random.default_rng(42)
    for k in unique_ks:
        users_k = np.where(cluster_labels == k)[0]
        users_k = [u for u in users_k if u in recommendations]
        if len(users_k) < 2:
            continue
        pairs = min(n_pairs, len(users_k) * (len(users_k) - 1) // 2)
        for _ in range(pairs):
            u, v = rng.choice(users_k, size=2, replace=False)
            scores.append(jaccard(set(recommendations[u]), set(recommendations[v])))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Собирает все метрики за каждый шаг симуляции.
    """

    def __init__(self, K: int = 5,
                 initial_mean: np.ndarray | None = None,
                 initial_cov:  np.ndarray | None = None):
        self.K            = K
        self.initial_mean = initial_mean
        self.initial_cov  = initial_cov
        self.history: list[dict] = []
        self._prev_mean   = None
        self._prev_cov    = None

    # ------------------------------------------------------------------
    def step(self, t: int, dataset,
             recommendations: dict,
             interactions: list,
             cluster_labels: np.ndarray | None = None,
             true_preference_matrix: np.ndarray | None = None,
             past_exposure_count: np.ndarray | None = None) -> dict:
        """Вычисляет и записывает все метрики для шага t."""
        m = {'t': t}
        U = dataset.users_embeddings
        I = dataset.items_embeddings
        n_items = I.shape[0] if I is not None else 1

        # ---- 1. Diversity & exposure ----
        if recommendations:
            recs_list = list(recommendations.values())

            # intra-list diversity
            divs = []
            for items in recs_list:
                if len(items) >= 2:
                    embs = I[np.array(items, dtype=int) % n_items]
                    dists = [np.linalg.norm(embs[i] - embs[j])
                             for i in range(len(embs))
                             for j in range(i + 1, len(embs))]
                    divs.append(float(np.mean(dists)))
            m['intra_list_diversity'] = float(np.mean(divs)) if divs else 0.0

            # catalog coverage
            shown = set(int(x) for v in recs_list for x in v)
            m['catalog_coverage'] = len(shown) / n_items

            # exposure entropy & gini
            exp_cnt = np.zeros(n_items)
            for v in recs_list:
                for x in v:
                    exp_cnt[int(x) % n_items] += 1
            total = exp_cnt.sum()
            if total > 0:
                p = exp_cnt / total
                p_nz = p[p > 0]
                m['exposure_entropy'] = float(scipy_entropy(p_nz))
                n = len(p)
                idx_sorted = np.argsort(p)
                cum = np.cumsum(p[idx_sorted])
                m['gini_exposure'] = float(
                    1 - 2 * cum[:-1].sum() / (n * p.sum() + 1e-12))
            else:
                m['exposure_entropy'] = 0.0
                m['gini_exposure']    = 0.0
        else:
            m.update(intra_list_diversity=0.0, catalog_coverage=0.0,
                     exposure_entropy=0.0, gini_exposure=0.0)

        # ---- 2. User geometry ----
        if U.shape[0] > 2:
            Sigma = np.cov(U, rowvar=False)
            Sigma = np.atleast_2d(Sigma)
            eigs  = np.linalg.eigvalsh(Sigma)
            m['trace_sigma']       = float(np.trace(Sigma))
            s, ld = np.linalg.slogdet(Sigma)
            m['log_det_sigma']     = float(ld) if s > 0 else -200.0
            m['leading_eigenvalue']= float(eigs[-1])
            m['min_eigenvalue']    = float(eigs[0])
        else:
            m.update(trace_sigma=0.0, log_det_sigma=-200.0,
                     leading_eigenvalue=0.0, min_eigenvalue=0.0)

        # ---- 3. Cluster metrics ----
        if cluster_labels is not None:
            uniq = np.unique(cluster_labels)
            centroids, intra_vars = [], []
            for k in uniq:
                uk = U[cluster_labels == k]
                if len(uk) > 1:
                    intra_vars.append(float(np.mean(np.var(uk, axis=0))))
                    centroids.append(np.mean(uk, axis=0))
            m['intra_cluster_variance'] = float(np.mean(intra_vars)) if intra_vars else 0.0

            if len(centroids) > 1:
                dists = [np.linalg.norm(centroids[i] - centroids[j])
                         for i in range(len(centroids))
                         for j in range(i + 1, len(centroids))]
                m['inter_cluster_distance'] = float(np.mean(dists))
            else:
                m['inter_cluster_distance'] = 0.0

            if U.shape[0] > len(uniq) >= 2:
                try:
                    m['silhouette_score'] = float(sk_silhouette(U, cluster_labels))
                except Exception:
                    m['silhouette_score'] = 0.0
            else:
                m['silhouette_score'] = 0.0

            if recommendations:
                m['intra_jaccard'] = mean_intra_cluster_jaccard(
                    recommendations, cluster_labels)
            else:
                m['intra_jaccard'] = 0.0
        else:
            m.update(intra_cluster_variance=0.0, inter_cluster_distance=0.0,
                     silhouette_score=0.0, intra_jaccard=0.0)

        # ---- 4. KL divergence ----
        if self.initial_mean is not None and U.shape[0] > 2:
            mean_t = np.mean(U, axis=0)
            cov_t  = np.cov(U, rowvar=False)
            m['kl_from_initial'] = kl_gaussian(
                mean_t, np.atleast_2d(cov_t),
                self.initial_mean, np.atleast_2d(self.initial_cov))

            if self._prev_mean is not None:
                m['kl_step'] = kl_gaussian(
                    mean_t, np.atleast_2d(cov_t),
                    self._prev_mean, np.atleast_2d(self._prev_cov))
            else:
                m['kl_step'] = 0.0
            self._prev_mean = mean_t.copy()
            self._prev_cov  = np.atleast_2d(cov_t).copy()
        else:
            m.update(kl_from_initial=0.0, kl_step=0.0)

        # ---- 5. Quality metrics ----
        n_recs = sum(len(v) for v in recommendations.values()) if recommendations else 1
        m['observed_quality'] = len(interactions) / max(n_recs, 1)

        if true_preference_matrix is not None and recommendations:
            m['true_quality'] = ndcg_at_k(
                recommendations, true_preference_matrix, self.K)
        else:
            m['true_quality'] = 0.0
        m['bias_gap'] = m['true_quality'] - m['observed_quality']

        # ---- 6. Exposure dependence (placeholder, computed in H6) ----
        m['exposure_dependence'] = 0.0

        self.history.append(m)
        return m

    # ------------------------------------------------------------------
    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def compute_collapse_time(self, threshold_ratio: float = 0.5,
                               metric: str = 'intra_list_diversity') -> int | None:
        """Шаг, на котором метрика впервые падает до threshold_ratio от начального значения."""
        df = self.get_dataframe()
        if metric not in df.columns or len(df) == 0:
            return None
        init_val  = df[metric].iloc[0]
        threshold = init_val * threshold_ratio
        below     = df[df[metric] < threshold]
        return int(below['t'].iloc[0]) if len(below) > 0 else None
