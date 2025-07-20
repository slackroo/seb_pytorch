import lightning as L
import torch
import torchmetrics
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(100, 50),
            torch.nn.BatchNorm1d(50),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(50, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits


class PyTorchMLP2(torch.nn.Module):
    def __init__(self, num_features, hidden_units, num_classes):
        super().__init__()

        all_layers = []
        for hidden_unit in hidden_units:
            layer = torch.nn.Linear(num_features, hidden_unit)
            all_layers.append(layer)
            all_layers.append(torch.nn.ReLU())
            num_features = hidden_unit
        output_layer = torch.nn.Linear(hidden_units[-1], num_classes)
        all_layers.append(output_layer)

        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.layers(x)
        return logits


def get_dataset_loaders():
    train_dataset = datasets.MNIST(
        root="./assets/mnist", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = datasets.MNIST(
        root="./assets/mnist", train=False, transform=transforms.ToTensor()
    )

    train_dataset, val_dataset = random_split(train_dataset, lengths=[55000, 5000],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def compute_accuracy(model, dataloader, device=None):
    if device is None:
        device = torch.device("cpu")
    model = model.eval()

    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./assets/mnist", batch_size=64, num_workers=0):
        super().__init__()

        # attributes we save and access diring calls
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        # access the train, val, test and predict data for Multi GPU processing

        self.mnist_test = datasets.MNIST(
            root=self.data_dir,
            transform=transforms.ToTensor(),
            train=False
        )
        self.mnist_predict = datasets.MNIST(
            self.data_dir,
            transform=transforms.ToTensor(),
            train=False
        )
        mnist_full = datasets.MNIST(
            self.data_dir,
            transform=transforms.ToTensor(),
            train=True
        )
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,

        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate, num_classes=2):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        # save settings and hhyperparams to the log dir
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

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
        self.log("train_loss", loss)

        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )

        return loss  # This is passed to optimizer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


class LightningModel2(L.LightningModule):
    def __init__(self, pytorch_model=None, hidden_units=None, learning_rate=None):
        super().__init__()
        self.learning_rate = learning_rate
        if pytorch_model is None:
            self.model = PyTorchMLP2(num_features=100, hidden_units=hidden_units, num_classes=2)
        else:
            self.model = pytorch_model

        if hidden_units is None:
            hidden_units = [100, 50]

        self.hidden_units = hidden_units

        # save settings and hhyperparams to the log dir
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

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
        self.log("train_loss", loss)

        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )

        return loss  # This is passed to optimizer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


class CustomDataset(Dataset):
    def __init__(self, feature_array, label_array, transform=None):
        self.x = feature_array
        self.y = label_array
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.y.shape[0]


class CustomDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./mnist", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: str):
        X, y = make_classification(
            n_samples=20000,
            n_features=100,
            n_informative=10,
            n_redundant=40,
            n_repeated=25,
            n_clusters_per_class=5,
            flip_y=0.05,
            class_sep=0.5,
            random_state=123,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1, shuffle=True
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=1, shuffle=True
        )

        self.train_dataset = CustomDataset(
            feature_array=X_train.astype(np.float32),
            label_array=y_train.astype(np.int64),
        )

        self.val_dataset = CustomDataset(
            feature_array=X_val.astype(np.float32), label_array=y_val.astype(np.int64)
        )

        self.test_dataset = CustomDataset(
            feature_array=X_test.astype(np.float32), label_array=y_test.astype(np.int64)
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=32, shuffle=False, num_workers=0
        )
        return test_loader

class Cifar10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path="./",
        batch_size=64,
        height_width=None,
        num_workers=0,
        train_transform=None,
        test_transform=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.height_width = height_width

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path, download=True)

        if self.height_width is None:
            self.height_width = (32, 32)

        if self.train_transform is None:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(self.height_width),
                    transforms.ToTensor(),
                ]
            )

        if self.test_transform is None:
            self.test_transform = transforms.Compose(
                [
                    transforms.Resize(self.height_width),
                    transforms.ToTensor(),
                ]
            )

        return

    def setup(self, stage=None):
        train = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.train_transform,
            download=False,
        )

        self.test = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.test_transform,
            download=False,
        )

        self.train, self.valid = random_split(train, lengths=[45000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader

def plot_loss_and_acc(
        log_dir, loss_ylim=(0.0, 0.9), acc_ylim=(0.7, 1.0), save_loss=None, save_acc=None
):
    metrics = pd.read_csv(f"{log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )

    plt.ylim(loss_ylim)
    if save_loss is not None:
        plt.savefig(save_loss)

    df_metrics[["train_acc", "val_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    plt.ylim(acc_ylim)
    if save_acc is not None:
        plt.savefig(save_acc)
