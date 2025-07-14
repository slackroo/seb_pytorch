from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from shared_utilities import CustomDataModule,LightningModel,PyTorchMLP
from watermark import watermark

if __name__ == '__main__':
    print(watermark(packages='torch,lightning', python=True))

    cli = LightningCLI(
        model_class=LightningModel,
        datamodule_class=CustomDataModule,
        run=False,
        save_config_callback=False,
        seed_everything_default=123,
        trainer_defaults={
            "max_epochs": 10,
            "accelerator": "cpu",
            "callbacks": [ModelCheckpoint(monitor="val_acc", mode="max")]
        },
    )

    pytorch_model = PyTorchMLP(num_features=100, num_classes=2)
    lighting_model = LightningModel(model=pytorch_model, learning_rate=cli.model.learning_rate)

    cli.trainer.fit(lighting_model, cli.datamodule)
    cli.trainer.test(datamodule=cli.datamodule)
