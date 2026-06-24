"""
Модели рекомендаций, совместимые с базовым кодом [m1p]base_code.ipynb.
"""

import torch
import torch.nn as nn


class RecModel(nn.Module):
    """
    Бинарный классификатор: (user_emb, item_emb) -> P(match).
    Архитектура идентична базовому коду.
    """

    def __init__(self, embedding_user_dim: int, embedding_item_dim: int,
                 hidden_size: int = 32):
        super().__init__()
        self.fc1     = nn.Linear(embedding_user_dim + embedding_item_dim, hidden_size)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        user : (B, user_dim)
        item : (B, item_dim)
        """
        x = torch.cat([user, item], dim=-1)
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x)).squeeze(-1)


class MFModel(nn.Module):
    """
    Бинарный bilinear-классификатор: score(u, v) = sigmoid(u^T M v + b).

    Геометрия скорa явно зависит от пары (u, v): при identity-инициализации
    `M = I` это в точности `sigmoid(<u, v>)`. Обучаемая матрица M ∈ R^{d×d}
    позволяет искажать метрику и подстраивать предпочтения. По числу
    параметров (d^2 + 1) кратно меньше MLP-классификатора.

    Сигнатура forward совпадает с RecModel, поэтому модель — drop-in
    замена в SimulationEnvironment._retrain и _make_recommendations.

    Parameters
    ----------
    embedding_user_dim, embedding_item_dim : int
        Должны совпадать (требование bilinear).
    identity_init : bool
        Если True, M инициализируется единичной матрицей (рекомендация в
        REPORT.md §7.1(a)). Если False — Xavier-нормально.
    """

    def __init__(self, embedding_user_dim: int, embedding_item_dim: int,
                 identity_init: bool = True):
        super().__init__()
        if embedding_user_dim != embedding_item_dim:
            raise ValueError('MFModel требует совпадения user/item dim')
        d = embedding_user_dim
        if identity_init:
            self.M = nn.Parameter(torch.eye(d))
        else:
            self.M = nn.Parameter(torch.empty(d, d))
            nn.init.xavier_normal_(self.M)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        # (B, d) @ (d, d) -> (B, d); * (B, d) -> sum -> (B,)
        score = (user @ self.M * item).sum(dim=-1) + self.b
        return torch.sigmoid(score)
