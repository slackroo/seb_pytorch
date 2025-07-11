import os
import torch
import watermark
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image

IMG_DIR = "./assets/mnist-pngs/"


class MyDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # based on df
        self.img_names = df['filepath']
        self.labels = df['label']

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def view_batch_images(batch):
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title("training image")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                batch[0][:64],
                padding=2,
                normalize=True
            ),
            (1, 2, 0)
        )
    )
    plt.show()


if __name__ == "__main__":
    print(watermark.watermark(packages="torch", python=True))

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop((28, 28)),
                transforms.ToTensor(),
                # normalize images to [-1, 1] range
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop((28, 28)),  # we n
                transforms.ToTensor(),
                # normalize images to [-1, 1] range
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    }

    train_dataset = MyDataset(
        csv_path=IMG_DIR + "/new_train.csv",
        img_dir=IMG_DIR,
        transform=data_transforms['train']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_dataset = MyDataset(
        csv_path=IMG_DIR + '/new_val.csv',
        img_dir=IMG_DIR,
        transform=data_transforms['test']
    )

    test_dataset = MyDataset(
        csv_path=IMG_DIR + "/test.csv",
        img_dir=IMG_DIR,
        transform=data_transforms['test']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    n_pochs = 1

    for epoch in range(n_pochs):

        for batch_idx , (x,y) in enumerate(train_loader):

            if batch_idx>3:
                break

            print("Batch Index: ", batch_idx, end="")
            print("| Batch Size:", y.shape[0], end="")
            print("| x shape:", x.shape, end="")
            print("| y shape:", y.shape)

    print("labels from current batch:", y)

    batch = next(iter(train_loader))
    # print(batch)
    view_batch_images(batch[0])
