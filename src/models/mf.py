import torch.nn as nn
from torch import Tensor


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim=n_factors)
        self.item_emb = nn.Embedding(n_items, embedding_dim=n_factors)
        # Initializing weights
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)

    def forward(self, user: Tensor, item: Tensor):
        # matrix multiplication
        u = self.user_emb(user)  # (n_users, emb_size)
        v = self.item_emb(item)  # (n_movies, emb_size)
        return (u * v).sum(1)  # (batch_size of ratings)
