from pyexpat import features
from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.utils.data import DataLoader
from watermark import watermark
from shared_utilities import PyTorchMLP, get_dataset_loaders, compute_accuracy
import lightning as L
import torchmetrics


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    # Shared step that is used in
    # training step, val step, and test step

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("Training loss", loss)

        self.train_accuracy(predicted_labels, true_labels)
        self.log(
            "train accuracy", self.train_accuracy, prog_bar=True, on_epoch=True, on_step=False
        )

        return loss  # This is passed to optimizer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val loss", loss, prog_bar=True)
        self.val_accuracy(predicted_labels, true_labels)
        self.log("val accuracy", self.val_accuracy, prog_bar=True)

    def test_step(self, batch,batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_accuracy(predicted_labels, true_labels)
        self.log("test accuracy", self.test_accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == '__main__':
    print(watermark(packages='torch', python=True))
    print(f"Torch CUDA available? {torch.cuda.is_available()}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = get_dataset_loaders()

    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)
    lighting_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices='auto',
        # deterministic=True # this the random seed constant setting.
    )
    trainer.fit(
        model=lighting_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    train_acc = trainer.test(dataloaders=train_loader)[0]['test accuracy']
    val_acc = trainer.test(dataloaders=val_loader)[0]['test accuracy']
    test_acc = trainer.test(dataloaders=test_loader)[0]['test accuracy']

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
