import numpy as np
from sklearn.cluster import KMeans


class GMMUserGenerator:
    """
    Генерирует пользователей из смеси гауссиан (K компонент).

    Уход: наименее активные пользователи (по числу взаимодействий
    в последние memory_effect шагов) покидают систему.
    Число уходящих = replacement_rate × n_users (фиксированная замена).
    Новые пользователи сэмплируются из GMM с параметрами, обновлёнными
    из текущей популяции (органический drift — без drift_alpha).
    """

    def __init__(self, component_means, component_covs, component_weights,
                 replacement_rate: float = 0.02, memory_effect: int = 6):
        self.component_means    = [np.array(m, dtype=float) for m in component_means]
        self.component_covs     = [np.array(c, dtype=float) for c in component_covs]
        self.component_weights  = np.array(component_weights, dtype=float)
        self.component_weights /= self.component_weights.sum()
        self.K                  = len(component_means)
        self.replacement_rate   = replacement_rate
        self.memory_effect      = memory_effect
        self.embeddings         = None
        self.cluster_labels     = None

    def initialize(self, n_users):
        counts = np.round(self.component_weights * n_users).astype(int)
        counts[-1] = n_users - counts[:-1].sum()
        parts, labels = [], []
        for k in range(self.K):
            n_k  = max(counts[k], 1)
            embs = np.random.multivariate_normal(
                self.component_means[k], self.component_covs[k], n_k)
            parts.append(embs)
            labels.extend([k] * n_k)
        self.embeddings     = np.vstack(parts)
        self.cluster_labels = np.array(labels, dtype=int)
        return self.embeddings.copy(), self.cluster_labels.copy()

    def _sample_gmm(self, n):
        if n == 0:
            dim = self.component_means[0].shape[0]
            return np.zeros((0, dim)), np.array([], dtype=int)
        k_choices = np.random.choice(self.K, size=n, p=self.component_weights)
        parts, labels = [], []
        for k in range(self.K):
            n_k = int(np.sum(k_choices == k))
            if n_k == 0:
                continue
            embs = np.random.multivariate_normal(
                self.component_means[k], self.component_covs[k], n_k)
            parts.append(embs)
            labels.extend([k] * n_k)
        if not parts:
            dim = self.component_means[0].shape[0]
            return np.zeros((0, dim)), np.array([], dtype=int)
        return np.vstack(parts), np.array(labels, dtype=int)

    def step(self, dataset, t):
        """
        Один шаг смены аудитории.

        Удаляем replacement_rate × n_users наименее активных пользователей.
        Добавляем столько же из текущего GMM.
        Популяция остаётся стационарной.

        Returns
        -------
        new_embeddings, new_cluster_labels, new_matrix, deleted_indices
        """
        n_users = self.embeddings.shape[0]
        n_replace = max(1, int(self.replacement_rate * n_users))
        n_replace = min(n_replace, n_users - 1)

        # Активность: число взаимодействий в последние memory_effect шагов
        activity  = np.nansum(dataset.matrix >= t - self.memory_effect, axis=1)
        deleted_idx = np.argsort(activity)[:n_replace]

        # Новые пользователи из текущего GMM
        new_embs, new_labels = self._sample_gmm(n_replace)

        keep_mask = np.ones(n_users, dtype=bool)
        keep_mask[deleted_idx] = False

        self.embeddings     = np.vstack([self.embeddings[keep_mask], new_embs])
        self.cluster_labels = np.concatenate([
            self.cluster_labels[keep_mask], new_labels])

        matrix   = np.delete(dataset.matrix, deleted_idx, axis=0)
        new_rows = np.full((n_replace, matrix.shape[1]), np.nan)
        matrix   = np.vstack([matrix, new_rows])

        return self.embeddings.copy(), self.cluster_labels.copy(), matrix, deleted_idx

    def update_component_params_from_data(self):
        """Обновляет μ_k, Σ_k из текущих эмбеддингов (органический дрейф)."""
        for k in range(self.K):
            mask = self.cluster_labels == k
            n_k  = np.sum(mask)
            if n_k > 2:
                self.component_means[k] = np.mean(self.embeddings[mask], axis=0)
                cov = np.cov(self.embeddings[mask], rowvar=False)
                self.component_covs[k]  = np.atleast_2d(cov)

    @property
    def distrib_params(self):
        return {'mean': self.component_means[0], 'cov': self.component_covs[0]}


class EmpiricalUserGenerator:
    """
    Генерирует пользователей из эмпирического распределения N(μ_t, Σ_t).

    Соответствует архитектуре [m1p]: новые пользователи сэмплируются
    из текущего распределения активных пользователей.
    Cluster labels определяются KMeans на текущей популяции.

    Parameters
    ----------
    replacement_rate : float
        Доля популяции, заменяемая за шаг.
    n_clusters : int
        Число кластеров для cluster_labels (KMeans).
    memory_effect : int
        Горизонт активности для отбора уходящих пользователей.
    """

    def __init__(self, replacement_rate: float = 0.02,
                 n_clusters: int = 3, memory_effect: int = 6):
        self.replacement_rate = replacement_rate
        self.n_clusters       = n_clusters
        self.memory_effect    = memory_effect
        self.K                = n_clusters
        self.embeddings       = None
        self.cluster_labels   = None
        self._cluster_centers = None

    def initialize(self, embeddings: np.ndarray):
        self.embeddings     = np.array(embeddings, dtype=float)
        self.cluster_labels = self._assign_clusters(self.embeddings)
        return self.embeddings.copy(), self.cluster_labels.copy()

    def _assign_clusters(self, embs: np.ndarray) -> np.ndarray:
        if self.n_clusters <= 1 or len(embs) < self.n_clusters:
            return np.zeros(len(embs), dtype=int)
        if self._cluster_centers is None:
            km = KMeans(n_clusters=self.n_clusters, n_init=5, random_state=0)
            km.fit(embs)
            self._cluster_centers = km.cluster_centers_.copy()
        dists  = np.linalg.norm(
            embs[:, None, :] - self._cluster_centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Экспоненциальное скользящее среднее для центров кластеров
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                self._cluster_centers[k] = (
                    0.9 * self._cluster_centers[k] +
                    0.1 * np.mean(embs[mask], axis=0))
        return labels.astype(int)

    def step(self, dataset, t):
        n_users   = self.embeddings.shape[0]
        n_replace = max(1, int(self.replacement_rate * n_users))
        n_replace = min(n_replace, n_users - 1)

        # Активность за последние memory_effect шагов
        activity    = np.nansum(dataset.matrix >= t - self.memory_effect, axis=1)
        deleted_idx = np.argsort(activity)[:n_replace]

        keep_mask = np.ones(n_users, dtype=bool)
        keep_mask[deleted_idx] = False
        survivors = self.embeddings[keep_mask]

        # Эмпирическое распределение выживших
        mu  = np.mean(survivors, axis=0)
        cov = np.cov(survivors, rowvar=False)
        cov = np.atleast_2d(cov) + np.eye(np.atleast_2d(cov).shape[0]) * 1e-6

        new_embs = np.random.multivariate_normal(mu, cov, n_replace)

        self.embeddings     = np.vstack([survivors, new_embs])
        self.cluster_labels = self._assign_clusters(self.embeddings)

        matrix   = np.delete(dataset.matrix, deleted_idx, axis=0)
        new_rows = np.full((n_replace, matrix.shape[1]), np.nan)
        matrix   = np.vstack([matrix, new_rows])

        return self.embeddings.copy(), self.cluster_labels.copy(), matrix, deleted_idx
