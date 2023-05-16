# import argparse
# import importlib
import warnings
from argparse import ArgumentParser

import lightning as L
import torch
from datasets.movielens import MovielensDataModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateFinder,
    ModelCheckpoint,
)
from lit_models.base import LightningModel
from models.mf_with_bias import MatrixFactorizationWithBias
from models.neu_mf import ConfigNeuMF, NeuMF
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.warnings import PossibleUserWarning

warnings.filterwarnings("ignore", category=PossibleUserWarning)

# DATA_CLASS_MODULE = "src.datasets"
# MODEL_CLASS_MODULE = "src.models"


# def import_class(module_and_class_name: str) -> type:
#     """Import class from a module, e.g. 'text_recognizer.models.MLP'."""
#     module_name, class_name = module_and_class_name.rsplit(".", 1)
#     module = importlib.import_module(module_name)
#     class_ = getattr(module, class_name)
#     return class_


# def setup_data_from_args(args: LazySettings):
#     data_class = import_class(args.trainer.logger.class_path)
#     data = data_class(**args.trainer.logger.init_args)
#     return data


# def setup_model_from_args(args: LazySettings):
#     model_class = import_class(args.model.pytorch_model)
#     lightning_model = model_class(**args.model.pytorch_model)
#     return lightning_model


def main(hparams) -> None:
    torch.manual_seed(123)

    # Load datamodules
    dm = MovielensDataModule(target=hparams.target, batch_size=hparams.batch_size)
    dm.setup()

    n_users, n_items = dm.n_users, dm.n_items
    print(n_users, n_items)

    # Instantiate pytorch model
    # pytorch_model = MatrixFactorizationWithBias(
    #     n_users=n_users, n_items=n_items, n_factors=hparams.emb_size
    # )
    config = ConfigNeuMF()
    pytorch_model = NeuMF(config)
    # pytorch_model = NCF(config)

    # Define pytroch model
    lightning_model = LightningModel(
        pytorch_model=pytorch_model, learning_rate=hparams.learning_rate
    )

    # Setup callbacks and trainer
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="min", monitor="valid_loss"),
        EarlyStopping(monitor="valid_loss", min_delta=0, patience=2),
        # LearningRateFinder(min_lr=0.01, max_lr=0.5),
    ]
    trainer = L.Trainer(
        callbacks=callbacks,
        max_epochs=hparams.epochs,
        accelerator="cpu",
        devices=1,
        deterministic=True,
        # logger=CSVLogger(save_dir="logs/", name=hparams.name, version=hparams.emb_size),
        logger=TensorBoardLogger(save_dir="lightning_logs", name="name"),
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=lightning_model, datamodule=dm)

    # TODO: Save in a file
    best_model_path = callbacks[0].best_model_path
    print(f"{best_model_path=}")

    # dataloaders = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
    # _ = trainer.validate(dataloaders=dataloaders, ckpt_path="best")
    # trainer.validate(dataloaders=dm.train_dataloader(), ckpt_path="best")
    # trainer.validate(datamodule=dm, ckpt_path="best")
    # trainer.test(datamodule=dm, ckpt_path="best")

    # TODO: Despues de entrenar calcular RMSE en el validation y test


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--config", default="configs/config2.yaml")
    # args = parser.parse_args()

    # settings = Dynaconf(root_path="configs", settings_files=[args.config])
    # data_class = import_class(settings.trainer.logger.class_path)
    # data_module = setup_data_from_args(settings)
    # lightning_model = setup_model_from_args(settings)

    # data_class = import_class(f"{DATA_CLASS_MODULE}.{settings.data_class}")
    # model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    parser = ArgumentParser()
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--batch_size", default=4096)
    parser.add_argument("--emb_size", default=32)
    parser.add_argument("--learning_rate", default=0.01)
    parser.add_argument("--target", default="rating")

    args = parser.parse_args()
    main(args)
