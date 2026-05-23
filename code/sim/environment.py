"""
Оркестратор симуляции replenishing recommender loop.

Архитектура:
- уход:   Бернулли(p_dep) на каждого пользователя;
- приход: семпл из N(mu_t, Sigma_t) по текущей популяции (органический drift);
- переобучение: каждые T_ret шагов в режиме closed_loop.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sim.user_generator import GMMUserGenerator, EmpiricalUserGenerator
from sim.click_model    import ClickModel
from sim.metrics        import MetricsTracker


class _PairDataset(Dataset):
    """(user_emb, item_emb, label) для обучения бинарного классификатора."""
    def __init__(self, users_emb, items_emb, matrix, subsample: int = 4000):
        pairs_pos, pairs_neg = [], []
        n_u, n_i = matrix.shape
        for u in range(n_u):
            pos_idx = np.where(~np.isnan(matrix[u]) & (matrix[u] >= 0.5))[0]
            neg_idx = np.where(~np.isnan(matrix[u]) & (matrix[u] < 0.5))[0]
            for i in pos_idx:
                pairs_pos.append((u, i, 1.0))
            for i in neg_idx:
                pairs_neg.append((u, i, 0.0))
        n = min(len(pairs_pos), len(pairs_neg), subsample // 2)
        if n == 0:
            n = max(len(pairs_pos), len(pairs_neg), 1)
        pos = pairs_pos[:n] if len(pairs_pos) >= n else pairs_pos
        neg = pairs_neg[:n] if len(pairs_neg) >= n else pairs_neg
        self.data      = pos + neg
        self.users_emb = users_emb.astype(np.float32)
        self.items_emb = items_emb.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, lbl = self.data[idx]
        return (torch.tensor(self.users_emb[u]),
                torch.tensor(self.items_emb[i]),
                torch.tensor(lbl, dtype=torch.float32))


class ExperimentDataset:
    """Контейнер состояния симуляции."""

    def __init__(self, users_embeddings, items_embeddings, rating_matrix):
        self.users_embeddings = np.array(users_embeddings, dtype=float)
        self.items_embeddings = np.array(items_embeddings, dtype=float)
        self.matrix           = np.array(rating_matrix, dtype=float)

    def update_ratings(self, interactions, t):
        n_u, n_i = self.matrix.shape
        for uid, iid, rating in interactions:
            uid, iid = int(uid), int(iid)
            if uid < n_u and iid < n_i:
                self.matrix[uid, iid] = float(t)


class SimulationEnvironment:
    """
    Оркестратор симуляции feedback loop в рекомендательной системе.

    Parameters
    ----------
    dataset          : ExperimentDataset
    rec_model        : nn.Module  — модель RecModel
    user_generator   : GMMUserGenerator | EmpiricalUserGenerator
    click_model      : ClickModel
    serving_policy   : ServingPolicy
    mode             : 'closed_loop' | 'static' | 'fresh_oracle' | 'no_influence'
    true_preference_matrix : np.ndarray | None
    retrain_period   : int
    K                : int   — топ-K рекомендаций
    device           : torch.device
    seen_filter      : bool  — если False, не фильтруем уже просмотренные объекты
                        (нужно для корректного измерения intra_jaccard)
    """

    def __init__(self, dataset: ExperimentDataset,
                 rec_model: nn.Module,
                 user_generator,
                 click_model: ClickModel,
                 serving_policy,
                 mode: str = 'closed_loop',
                 true_preference_matrix: np.ndarray | None = None,
                 retrain_period: int = 10,
                 K: int = 10,
                 device=None,
                 initial_mean: np.ndarray | None = None,
                 initial_cov: np.ndarray | None = None,
                 fresh_oracle_fn=None,
                 seen_filter: bool = True,
                 user_drift_beta: float = 0.02,
                 drift_alpha: float = 0.0):

        self.dataset        = dataset
        self.rec_model      = rec_model
        self.user_gen       = user_generator
        self.click_model    = click_model
        self.serving        = serving_policy
        self.mode           = mode
        self.true_pref      = true_preference_matrix
        self.retrain_period = retrain_period
        self.K              = K
        self.device         = device or torch.device('cpu')
        self.fresh_oracle_fn = fresh_oracle_fn
        self.seen_filter      = seen_filter
        self.user_drift_beta  = user_drift_beta
        self.drift_alpha      = drift_alpha

        n_items = dataset.items_embeddings.shape[0]
        self.exposure_count = np.zeros(n_items, dtype=float)

        if initial_mean is None:
            initial_mean = np.mean(dataset.users_embeddings, axis=0)
        if initial_cov is None:
            initial_cov = np.cov(dataset.users_embeddings, rowvar=False)

        self.metrics = MetricsTracker(
            K=K,
            initial_mean=initial_mean,
            initial_cov=np.atleast_2d(initial_cov))

        self.rec_model = self.rec_model.to(self.device)

    def step(self, t: int) -> dict:
        """Один шаг симуляции: serve -> clicks -> update -> retrain -> metrics."""
        dataset = self.dataset
        U = dataset.users_embeddings
        I = dataset.items_embeddings
        n_users, n_items = U.shape[0], I.shape[0]

        # 1. Рекомендации
        recommendations = self._make_recommendations(U, I, n_users, n_items)

        # 2. Клики
        interactions = self._simulate_clicks(recommendations, U, n_users, n_items, t)

        # 3. Обновляем матрицу
        dataset.update_ratings(interactions, t)

        # Обновляем счётчики экспозиций
        for items in recommendations.values():
            for it in items:
                if it < len(self.exposure_count):
                    self.exposure_count[it] += 1

        # 4a. Preference drift: u_i <- (1-beta) u_i + beta v_{j(i)}.
        if self.user_drift_beta > 0 and interactions:
            U = dataset.users_embeddings
            I = dataset.items_embeddings
            n_u, n_i = U.shape[0], I.shape[0]
            for uid, iid, _ in interactions:
                uid, iid = int(uid), int(iid)
                if uid < n_u and iid < n_i:
                    U[uid] = ((1 - self.user_drift_beta) * U[uid] +
                               self.user_drift_beta * I[iid])
            dataset.users_embeddings = U
            # синхронизация массива у генератора, иначе user_gen.step()
            # перезапишет задрейфованные эмбеддинги
            if hasattr(self.user_gen, 'embeddings') and \
               self.user_gen.embeddings is not None:
                self.user_gen.embeddings = U.copy()

        # 4b. Drift центров GMM к средней позиции потреблённых объектов.
        if self.drift_alpha > 0 and interactions and hasattr(self.user_gen, 'component_means'):
            I = dataset.items_embeddings
            interacted_iids = [int(iid) for _, iid, _ in interactions
                               if int(iid) < I.shape[0]]
            if interacted_iids:
                mean_interacted = np.mean(I[interacted_iids], axis=0)
                for k in range(self.user_gen.K):
                    self.user_gen.component_means[k] = (
                        (1 - self.drift_alpha) * self.user_gen.component_means[k] +
                        self.drift_alpha * mean_interacted)

        # 4c. Генерация новых пользователей; параметры GMM пересчитываются
        # из текущих эмбеддингов (органический drift).
        if hasattr(self.user_gen, 'update_component_params_from_data'):
            self.user_gen.update_component_params_from_data()

        new_embs, new_labels, new_matrix, _ = self.user_gen.step(dataset, t)
        dataset.users_embeddings = new_embs
        dataset.matrix           = new_matrix

        # 5. Переобучение
        if self.mode == 'closed_loop' and t % self.retrain_period == 0:
            self._retrain(dataset)
        elif self.mode == 'fresh_oracle' and t % self.retrain_period == 0:
            if self.fresh_oracle_fn is not None:
                fresh_interactions = self.fresh_oracle_fn()
                self._retrain_on_interactions(dataset, fresh_interactions)
            else:
                self._retrain_oracle(dataset)

        # 6. Метрики
        metrics = self.metrics.step(
            t=t,
            dataset=dataset,
            recommendations=recommendations,
            interactions=interactions,
            cluster_labels=new_labels,
            true_preference_matrix=self.true_pref,
            past_exposure_count=self.exposure_count,
        )
        return metrics

    def _make_recommendations(self, U, I, n_users, n_items):
        """Формирует рекомендации через serving_policy."""
        self.rec_model.eval()
        recs = {}
        with torch.no_grad():
            u_t = torch.tensor(U, dtype=torch.float32).to(self.device)
            i_t = torch.tensor(I, dtype=torch.float32).to(self.device)
            for uid in range(n_users):
                u_rep  = u_t[uid].unsqueeze(0).expand(n_items, -1)
                scores = self.rec_model(u_rep, i_t).squeeze().cpu().numpy()
                if self.seen_filter:
                    seen = np.where(~np.isnan(self.dataset.matrix[uid]))[0]
                    scores[seen] = -np.inf
                items = self.serving.select(scores, self.K,
                                            exposure_count=self.exposure_count)
                if items is not None and len(items) > 0:
                    recs[uid] = items
        return recs

    def _simulate_clicks(self, recommendations, U, n_users, n_items, t):
        interactions = []
        for uid, items in recommendations.items():
            if uid >= n_users:
                continue
            if self.mode == 'no_influence':
                if self.true_pref is not None and uid < self.true_pref.shape[0]:
                    ts = self.true_pref[uid, [min(i, n_items - 1) for i in items]]
                    clicks = self.click_model.generate_clicks(
                        uid, items, ts, t,
                        past_exposure=self.exposure_count[
                            np.array(items) % len(self.exposure_count)])
                else:
                    clicks = []
            else:
                if self.true_pref is not None and uid < self.true_pref.shape[0]:
                    ts = np.array([
                        self.true_pref[uid, min(i, self.true_pref.shape[1] - 1)]
                        for i in items])
                else:
                    ts = np.random.uniform(0.3, 0.7, len(items))
                clicks = self.click_model.generate_clicks(
                    uid, items, ts, t,
                    past_exposure=self.exposure_count[
                        np.array(items) % len(self.exposure_count)])
            interactions.extend(clicks)
        return interactions

    def _retrain(self, dataset, n_epochs: int = 1, lr: float = 3e-4,
                  subsample: int = 4000):
        train_set = _PairDataset(
            dataset.users_embeddings, dataset.items_embeddings,
            dataset.matrix, subsample=subsample)
        if len(train_set) == 0:
            return
        loader    = DataLoader(train_set, batch_size=256, shuffle=True)
        optimizer = optim.Adam(self.rec_model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        self.rec_model.train()
        for _ in range(n_epochs):
            for u_b, i_b, lbl_b in loader:
                u_b   = u_b.to(self.device)
                i_b   = i_b.to(self.device)
                lbl_b = lbl_b.to(self.device)
                optimizer.zero_grad()
                pred  = self.rec_model(u_b, i_b)
                loss  = criterion(pred, lbl_b)
                loss.backward()
                optimizer.step()

    def _retrain_oracle(self, dataset, n_epochs: int = 1):
        if self.true_pref is None:
            return
        n_u = min(dataset.users_embeddings.shape[0], self.true_pref.shape[0])
        n_i = min(dataset.items_embeddings.shape[0], self.true_pref.shape[1])
        clean_matrix = np.full(dataset.matrix.shape, np.nan)
        clean_matrix[:n_u, :n_i] = self.true_pref[:n_u, :n_i]
        tmp_ds = ExperimentDataset(
            dataset.users_embeddings, dataset.items_embeddings, clean_matrix)
        self._retrain(tmp_ds, n_epochs=n_epochs)

    def _retrain_on_interactions(self, dataset, interactions, n_epochs: int = 1):
        if not interactions:
            return
        matrix = np.full(dataset.matrix.shape, np.nan)
        n_u, n_i = matrix.shape
        for uid, iid, rating in interactions:
            uid, iid = int(uid), int(iid)
            if uid < n_u and iid < n_i:
                matrix[uid, iid] = float(rating)
        tmp_ds = ExperimentDataset(
            dataset.users_embeddings, dataset.items_embeddings, matrix)
        self._retrain(tmp_ds, n_epochs=n_epochs)
