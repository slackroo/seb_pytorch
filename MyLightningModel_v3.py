import torch
from watermark import watermark
from shared_utilities import PyTorchMLP, MNISTDataModule, LightningModel
import lightning as L

if __name__ == '__main__':
    print(watermark(packages='torch', python=True))
    print(f"Torch CUDA available? {torch.cuda.is_available()}")

    torch.manual_seed(123)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dm = MNISTDataModule()

    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)
    lighting_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices='auto',
        deterministic=True, # this the random seed constant setting.
    )
    # trainer.fit(
    #     model=lighting_model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=val_loader
    # )

    trainer.fit(
        model=lighting_model,
        datamodule=dm
    )

    train_acc = trainer.validate(dataloaders=dm.train_dataloader())[0]['val accuracy']
    val_acc = trainer.validate(dataloaders=dm)[0]['val accuracy']
    test_acc = trainer.test(dataloaders=dm)[0]['test accuracy']

    print(
        f"Train Acc {train_acc * 100:.2f}%"
        f" | Val Acc {val_acc * 100:.2f}%"
        f" | Test Acc {test_acc * 100:.2f}%"
    )

PATH = "lightning.pt"
torch.save(pytorch_model.state_dict(), PATH)

# To load model:
# model = PyTorchMLP(num_features=784, num_classes=10)
# model.load_state_dict(torch.load(PATH))
# model.eval()
