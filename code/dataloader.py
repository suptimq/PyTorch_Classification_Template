import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as tvtrans

import pdb


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, train=True, transform=None, target_transform=None):
        self.root_dir = Path(dataset_path['root_path'])
        self.train_filepath = self.root_dir / dataset_path['train_filepath']
        self.test_filepath = self.root_dir / dataset_path['test_filepath']
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train is True:
            self.data_filepath = self.train_filepath
        else:
            self.data_filepath = self.test_filepath

        with h5py.File(self.data_filepath, 'r') as f:
            image_ds = f['image']
            self.images = image_ds[:, ]
            self.images = self.images[:, :, :, np.newaxis]
            label_ds = f['label']
            self.labels = label_ds[:]
            self.labels = self.labels[:, np.newaxis]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cur_images = self.images[idx, :, :, :]
        cur_labels = self.labels[idx, :]
        cur_labels = cur_labels.squeeze().astype(dtype=np.int64)

        # pdb.set_trace()

        if self.transform is not None:
            cur_images = self.transform(cur_images)

        if self.target_transform is not None:
            cur_labels = self.target_transform(cur_labels)

        return cur_images, cur_labels


# use to test the dataset class


def test_class():

    # Parameters for dataset
    dataset_path = {
        'root_path': '../data',
        'meta_filepath': None,
        'train_filepath': 'train.hdf5',
        'test_filepath': 'test.hdf5'
    }

    transform = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.RandomHorizontalFlip(p=0.5),
        tvtrans.RandomVerticalFlip(p=0.5),
        tvtrans.ToTensor()
    ])

    train_ds = Dataset(dataset_path, train=True, transform=transform)

    dl = torch.utils.data.DataLoader(train_ds,
                                     batch_size=4,
                                     shuffle=False)
    data_iter = iter(dl)
    for _ in range(2):
        images, labels = data_iter.next()

        f, axarr = plt.subplots(2, 2)
        j = 0
        for row in range(2):
            for col in range(2):
                cur_img = images[j, ]
                cur_label = labels[j]
                axarr[row, col].imshow(np.transpose(
                    cur_img, (1, 2, 0)).squeeze())
                axarr[row, col].set_title('{0}'.format(cur_label.item()))
                j += 1
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_class()
