import numpy as np


class ClickModel:
    """
    Параметризованная модель поведения пользователя.

    Parameters
    ----------
    adherence : float  (α ∈ [0,1])
        Насколько пользователь смещает выбор к рекомендованному.
        0 = полностью игнорирует рекомендации, 1 = кликает только на них.
    usage_rate : float  (ρ ∈ [0,1])
        Вероятность взаимодействия на каждом шаге.
    novelty_preference : float  (ν ∈ [0,1])
        Тяга к айтемам, которые пользователь ещё не видел.
    noise_level : float  (σ ≥ 0)
        Стандартное отклонение гауссовского шума в click probability.
    """

    def __init__(self, adherence: float = 0.7, usage_rate: float = 0.8,
                 novelty_preference: float = 0.3, noise_level: float = 0.1):
        self.adherence          = float(np.clip(adherence, 0, 1))
        self.usage_rate         = float(np.clip(usage_rate, 0, 1))
        self.novelty_preference = float(np.clip(novelty_preference, 0, 1))
        self.noise_level        = float(max(noise_level, 0))

    # ------------------------------------------------------------------
    def generate_clicks(self, user_id: int, recommended_items: np.ndarray,
                        true_scores: np.ndarray, t: int,
                        past_exposure: np.ndarray | None = None
                        ) -> list[tuple[int, int, float]]:
        """
        Генерирует список взаимодействий пользователя с рекомендованными айтемами.

        Parameters
        ----------
        user_id          : индекс пользователя
        recommended_items: массив индексов рекомендованных айтемов (длина K)
        true_scores      : истинная utility для каждого айтема (длина K)
        t                : текущий шаг симуляции
        past_exposure    : количество прошлых показов для каждого айтема
                           (len = K); если None — считается нулём

        Returns
        -------
        list of (user_id, item_id, observed_rating)
        """
        if np.random.random() > self.usage_rate:
            return []

        if past_exposure is None:
            past_exposure = np.zeros(len(recommended_items))

        interactions = []
        for idx, item_id in enumerate(recommended_items):
            ts = float(true_scores[idx]) if idx < len(true_scores) else 0.5

            # Novelty bonus: снижаем привлекательность часто показываемых айтемов
            novelty_bonus = self.novelty_preference * np.exp(
                -0.1 * past_exposure[idx]) if self.novelty_preference > 0 else 0.0

            # observed_rating = α·exposure_signal + (1-α)·true_utility + ν·novelty + ε
            exposure_signal = 1.0  # рекомендованному айтему — максимальная экспозиция
            observed = (self.adherence * exposure_signal +
                        (1 - self.adherence) * ts +
                        novelty_bonus +
                        np.random.normal(0, self.noise_level))
            observed = float(np.clip(observed, 0.0, 1.0))

            if np.random.random() < observed:
                interactions.append((user_id, int(item_id), observed))

        return interactions
