# Customize Dataset
import pandas as pd
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from helper import labelConverter


class SignalDataset(Dataset):
    def __init__(self, data, label, transform=None):
        """
        Args:
            data (Numpy Array format):  Numpy array with only signal samples
            label (Numpy Array format): Numpy array with all the labels
            transform:                  Transform methods
        """
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        signal = self.data[idx]                       # Shape (1024, 2)
        label = self.label[idx]

        # Get channels' mean and std values
        mean = np.mean(signal, axis=0)
        std = np.std(signal, axis=0)
        signal_reshape = signal[:, np.newaxis, :]     # Shape (1024, 1, 2)

        if self.transform:
            t = transforms.Compose([
                transforms.ToTensor(),                 # Convert (H, W, C) to (C, H, W)
                transforms.Normalize(mean, std)
            ])
            signal_norm = t(signal_reshape)            # Shape (2, 1024, 1)
            tensor_signal = signal_norm.squeeze(-1)    # Shape (2, 1024)
        else:
            # set_trace()
            # Input (1024, 2) => (2, 1024)
            tensor_signal = torch.from_numpy(signal).transpose(0, 1).float()

        tensor_label = torch.from_numpy(np.asarray(label)).float()

        sample = {'signal': tensor_signal, 'label': tensor_label}

        return sample


dataset_filename = './datasets/data.hdf5'
trainlabel_filename = './datasets/train_labels.csv'
aug = False

# Import data from HDF5
f = h5py.File(dataset_filename, 'r')
train_data = f['train']
test_data = f['test']

# Read train dataset labels
train_labels = pd.read_csv(trainlabel_filename)

# Convert the original labels to the numerical format
classes = [
    'FM', 'OQPSK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', 'AM-DSB-SC',
    'QPSK', 'OOK'
]
label_dict, label_dict_reverse = labelConverter(classes)

# Add the numerical labels column to the original DataFrame
train_labels['label'] = train_labels.apply(
    lambda row: label_dict[row.Category], axis=1)

train_data_np = train_data[: 10000]
train_labels_np = train_labels['label'].values[: 10000]

if aug:
    # Data Augmentation
    train_data_np_flip = np.flip(train_data_np, axis=1)
    train_data_total = np.concatenate((train_data_np, train_data_np_flip))

    train_labels_total = np.concatenate((train_labels_np, train_labels_np))
else:
    # Variable name consistency
    train_data_total = train_data_np.copy()
    train_labels_total = train_labels_np.copy()
