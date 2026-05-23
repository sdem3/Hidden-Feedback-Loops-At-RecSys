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
