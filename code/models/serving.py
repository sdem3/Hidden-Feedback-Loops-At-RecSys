"""
ServingPolicy — преобразует скоры модели в топ-K рекомендаций.
"""

import numpy as np


class ServingPolicy:
    """
    Режимы выдачи рекомендаций.

    Parameters
    ----------
    mode : str
        'top_k'          — детерминированный argmax (максимальное смещение)
        'epsilon_greedy' — ε-greedy: с вероятностью ε случайный айтем
        'softmax'        — пропорциональное сэмплирование softmax(scores/T)
        'diversity_aware'— штраф −λ·freq(item) к скору
        'ips_corrected'  — скоры делятся на propensity (пропорциональную частоте)
    epsilon : float       — ε для epsilon_greedy
    temperature : float   — T для softmax
    diversity_lambda : float — λ для diversity_aware
    """

    def __init__(self, mode: str = 'top_k',
                 epsilon: float = 0.1,
                 temperature: float = 1.0,
                 diversity_lambda: float = 0.5):
        assert mode in ('top_k', 'epsilon_greedy', 'softmax',
                        'diversity_aware', 'ips_corrected'), \
            f"Unknown mode: {mode}"
        self.mode             = mode
        self.epsilon          = epsilon
        self.temperature      = temperature
        self.diversity_lambda = diversity_lambda

    # ------------------------------------------------------------------
    def select(self, scores: np.ndarray, K: int,
               exposure_count: np.ndarray | None = None) -> np.ndarray:
        """
        Выбирает K айтемов по scores.

        Parameters
        ----------
        scores         : (n_items,)  — скоры модели; -inf для уже виденных
        K              : число рекомендаций
        exposure_count : (n_items,)  — накопленные счётчики показов (для diversity/IPS)

        Returns
        -------
        np.ndarray  shape (K,) — индексы рекомендованных айтемов
        """
        n = len(scores)
        valid = np.isfinite(scores)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return np.array([], dtype=int)
        K_eff = min(K, n_valid)

        if self.mode == 'top_k':
            return self._top_k(scores, K_eff)

        elif self.mode == 'epsilon_greedy':
            return self._epsilon_greedy(scores, K_eff)

        elif self.mode == 'softmax':
            return self._softmax_sample(scores, K_eff)

        elif self.mode == 'diversity_aware':
            return self._diversity_aware(scores, K_eff, exposure_count)

        elif self.mode == 'ips_corrected':
            return self._ips_corrected(scores, K_eff, exposure_count)

        return self._top_k(scores, K_eff)

    # ------------------------------------------------------------------
    def _top_k(self, scores, K):
        return np.argsort(scores)[::-1][:K]

    def _epsilon_greedy(self, scores, K):
        n      = len(scores)
        valid  = np.where(np.isfinite(scores))[0]
        chosen = []
        for _ in range(K):
            remaining = [v for v in valid if v not in chosen]
            if not remaining:
                break
            if np.random.random() < self.epsilon:
                chosen.append(int(np.random.choice(remaining)))
            else:
                best = max(remaining, key=lambda v: scores[v])
                chosen.append(best)
        return np.array(chosen, dtype=int)

    def _softmax_sample(self, scores, K):
        valid    = np.where(np.isfinite(scores))[0]
        s        = scores[valid]
        s        = s - s.max()  # numerical stability
        probs    = np.exp(s / max(self.temperature, 1e-6))
        probs   /= probs.sum()
        chosen   = np.random.choice(valid, size=min(K, len(valid)),
                                    replace=False, p=probs)
        return chosen

    def _diversity_aware(self, scores, K, exposure_count):
        valid = np.where(np.isfinite(scores))[0]
        if exposure_count is None:
            return self._top_k(scores, K)
        ec    = exposure_count[:len(scores)]
        ec_norm = ec / (ec.max() + 1e-9)
        adj_scores = scores.copy()
        adj_scores[valid] -= self.diversity_lambda * ec_norm[valid]
        return self._top_k(adj_scores, K)

    def _ips_corrected(self, scores, K, exposure_count):
        valid = np.where(np.isfinite(scores))[0]
        if exposure_count is None:
            return self._top_k(scores, K)
        ec    = exposure_count[:len(scores)]
        propensity = (ec + 1) / (ec.max() + 2)  # smoothed propensity
        adj_scores = scores.copy()
        adj_scores[valid] = scores[valid] / propensity[valid]
        return self._top_k(adj_scores, K)
