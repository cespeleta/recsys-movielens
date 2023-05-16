import torch.nn as nn
from torch import Tensor


class MatrixFactorizationWithBias(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int):
        super(MatrixFactorizationWithBias, self).__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim=n_factors)
        self.user_bias = nn.Embedding(n_users, embedding_dim=1)
        self.item_emb = nn.Embedding(n_items, embedding_dim=n_factors)
        self.item_bias = nn.Embedding(n_items, embedding_dim=1)
        # Initializing weights
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

    def forward(self, user: Tensor, item: Tensor):
        u = self.user_emb(user)
        v = self.item_emb(item)
        u_bias = self.user_bias(user).squeeze()
        v_bias = self.item_bias(item).squeeze()
        return (u * v).sum(1) + u_bias + v_bias  # len = batch_size
