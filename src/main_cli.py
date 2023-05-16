from pathlib import Path

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger
from lit_models.base import LightningModel


def cli_main():
    cli = LightningCLI(
        model_class=LightningModel,
        # save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        trainer_defaults={
            "devices": 1,
            "logger": TensorBoardLogger(
                save_dir="lightning_logs", name="default", sub_dir="tf_logs"
            ),
        },
        run=False,
    )
    # print(f"{cli.trainer.logger.name=}")
    # print(f"{cli.model.pytorch_model=}")

    print("Loading n_users and n_items from its DataModule")
    dm = cli.datamodule
    dm.setup()
    print(f"{dm.n_users=}, {dm.n_items=}")

    # Start training the model
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # Evaluate datasets with best model and save metrics a file
    dataloaders = [
        cli.datamodule.train_dataloader(),
        cli.datamodule.val_dataloader(),
        cli.datamodule.test_dataloader(),
    ]
    best = cli.trainer.validate(cli.model, dataloaders=dataloaders, ckpt_path="best")

    # Save best model metrics
    checkpoints_path = Path(cli.trainer.checkpoint_callback.best_model_path).parent
    print(f"best_model_path: {checkpoints_path}")

    with open(checkpoints_path / "training_metrics.csv", "w") as f:
        f.write(f"{best}")


if __name__ == "__main__":
    cli_main()

    # python src/main_cli.py --config configs/config_mf.yaml
