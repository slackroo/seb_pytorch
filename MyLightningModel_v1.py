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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        self.log("Training loss", loss)

        predicted_labels = torch.argmax(logits, dim=1)
        self.train_accuracy(predicted_labels, true_labels)
        self.log(
            "train accuracy", self.train_accuracy, prog_bar=True, on_epoch=True, on_step=False
        )

        return loss  # This is passed to optimizer for training

    def validation_step(self, batch, batch_idx):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        self.log("val loss", loss, prog_bar=True)

        predicted_labels = torch.argmax(logits, dim=1)
        self.val_accuracy(predicted_labels, true_labels)
        self.log(
            "val accuracy", self.val_accuracy, prog_bar=True
        )

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
    )
    trainer.fit(
        model=lighting_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

PATH = "lightning.pt"
torch.save(pytorch_model.state_dict(), PATH)

# To load model:
# model = PyTorchMLP(num_features=784, num_classes=10)
# model.load_state_dict(torch.load(PATH))
# model.eval()
