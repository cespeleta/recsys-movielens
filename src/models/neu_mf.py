from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ConfigNeuMF:
    num_users: int = 943
    num_items: int = 1625
    latent_dim_mf: int = 32
    latent_dim_mlp: int = 32
    dropout_rate_mf: float = 0.0
    dropout_rate_mlp: float = 0.0
    layers: list[int] = field(default_factory=lambda: [64, 32, 16, 8])


class NeuMF(nn.Module):
    def __init__(self, config: ConfigNeuMF):
        super(NeuMF, self).__init__()
        self.config = config
        self.dropout = config.dropout_rate_mf
        num_layers = len(config.layers)

        # Matrix Factorization embeddings
        self.user_emb_mf = nn.Embedding(config.num_users, config.latent_dim_mf)
        self.item_emb_mf = nn.Embedding(config.num_items, config.latent_dim_mf)

        # MLP embeddings
        self.user_emb_mlp = nn.Embedding(
            config.num_users, config.latent_dim_mf * (2 ** (num_layers - 1))
        )
        self.item_emb_mlp = nn.Embedding(
            config.num_items, config.latent_dim_mf * (2 ** (num_layers - 1))
        )

        # MLP layers
        mlp_modules = []
        for i in range(num_layers):  # [0, 1, 2] = len=3
            input_size = config.latent_dim_mf * (2 ** (num_layers - i))
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

        # predict_size = config.latent_dim_mf * 2
        self.predict_layer = nn.Linear(input_size, 1)

        # Initialize weights
        self._init_weights()

    def forward(self, user, item):
        # Matrix Factorization
        user_emb_mf = self.user_emb_mf(user)
        item_emb_mf = self.item_emb_mf(item)
        output_mf = user_emb_mf * item_emb_mf

        # MLP
        user_emb_mlp = self.user_emb_mlp(user)
        item_emb_mlp = self.item_emb_mlp(item)
        interaction = torch.cat((user_emb_mlp, item_emb_mlp), dim=-1)
        output_mlp = self.mlp_layers(interaction)

        # Concatenate MF and MLP outpus
        concat = torch.cat((output_mf, output_mlp), dim=-1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def _init_weights(self):
        nn.init.normal_(self.user_emb_mf.weight, std=0.01)
        nn.init.normal_(self.item_emb_mf.weight, std=0.01)
        nn.init.normal_(self.user_emb_mlp.weight, std=0.01)
        nn.init.normal_(self.item_emb_mlp.weight, std=0.01)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


if __name__ == "__main__":
    config = ConfigNeuMF()
    print(config)
    NeuMF(config)
