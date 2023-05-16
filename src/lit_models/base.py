from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanSquaredError


class LightningModel(LightningModule):
    """Generic Lightning class that must be initialized with a PyTorch module."""

    def __init__(
        self,
        pytorch_model: nn.Module,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pytorch_model = pytorch_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["pytorch_model"])

        # TODO: pasar por parametro la metrica
        self.train_metric = MeanSquaredError(squared=False)  # RMSE
        self.valid_metric = MeanSquaredError(squared=False)
        self.test_metric = MeanSquaredError(squared=False)

    def forward(self, users, items):
        return self.pytorch_model(users, items)

    def _shared_step(self, batch, batch_idx: int, prefix: str):
        users, items, true_ratings = batch["user"], batch["item"], batch["rating"]
        predicted_ratings = self(users, items)
        loss = F.mse_loss(predicted_ratings, true_ratings)
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        return loss, true_ratings, predicted_ratings

    def training_step(self, batch, batch_idx):
        loss, true_ratings, predicted_ratings = self._shared_step(
            batch, batch_idx, "train"
        )
        self.train_metric(predicted_ratings, true_ratings)
        self.log(
            "train_rmse", self.train_metric, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, true_ratings, predicted_ratings = self._shared_step(
            batch, batch_idx, "valid"
        )
        self.valid_metric(predicted_ratings, true_ratings)
        self.log(
            "valid_rmse", self.valid_metric, prog_bar=True, on_epoch=True, on_step=False
        )

    def test_step(self, batch, batch_idx):
        _, true_ratings, predicted_labels = self._shared_step(batch, batch_idx, "test")
        self.test_metric(predicted_labels, true_ratings)
        self.log("test_rmse", self.test_metric, prog_bar=False)

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        users, items = batch["user"], batch["item"]
        y_hat = self(users, items)
        return F.sigmoid(y_hat)  # add sigmoid

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=self.learning_rate, momentum=0.5
        # )
        return optimizer
