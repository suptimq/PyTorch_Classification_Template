import os
import h5py
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import pdb

import torch
import torchvision.transforms as tvtrans

from model import Net
from helper import init_model

# python inference.py --modelType Net --loading_epoch 190 --timestamp Jun04_18-18-46
parser = argparse.ArgumentParser()
parser.add_argument('--modelType',
                    default='Net',
                    help='model used for training')
parser.add_argument('--loading_epoch',
                    type=int,
                    required=True,
                    help='xth model loaded for inference')
parser.add_argument('--timestamp', required=True, help='model timestamp')
parser.add_argument('--outdim', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--num_sample', type=int, default=10,
                    help='number of testing samples')
opt = parser.parse_args()
print(opt)

root_path = Path(os.path.abspath(__file__))
code_root_path = root_path.parent
dataset_root_path = root_path.parents[1] / 'data'
image_root_path = dataset_root_path / 'images'

test_filepath = 'test.hdf5'

# Model
current_time = opt.timestamp
model_type_time = opt.modelType + '_{}'.format(current_time)

model_path = code_root_path / 'models' / model_type_time
model_filename = '{0}_model_ep{1:03}'
model_fullpath = str(model_path / model_filename)

model_para = {
    'model_type': opt.modelType,
    'outdim': opt.outdim,
    'model_path': model_path,
    'model_filename': model_filename,
    'loading_epoch': opt.loading_epoch,
    'gpu': opt.cuda
}

dataset_para = {
    'dataset_root_path': dataset_root_path,
    'test_filepath': test_filepath
}


def load_data_h5py(dataset_para):
    f = h5py.File(dataset_para['dataset_root_path'] /
                  dataset_para['test_filepath'], 'r')
    image_ds = f['image']
    images = image_ds[:, ]
    label_ds = f['label']
    labels = label_ds[:]
    return images, labels


def load_model(model_para):
    model = init_model(model_para)
    if model_para['gpu']:
        checkpoint = torch.load(model_fullpath.format(
            model_para['model_type'], model_para['loading_epoch']))
    else:
        checkpoint = torch.load(model_fullpath.format(
            model_para['model_type'], model_para['loading_epoch']),  map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == "__main__":
    # Import data from HDF5
    images, labels = load_data_h5py(dataset_para)

    test_transform = tvtrans.Compose([
        tvtrans.ToPILImage(),
        tvtrans.ToTensor(),
        tvtrans.Normalize((0.5, ), (0.5, ))
    ])

    # Load model
    model = load_model(model_para)

    print("LOADING MODEL FINISHED")
    print("INFERENCE START")

    correct_counts = 0
    total_counts = 0

    for _ in range(opt.num_sample):
        idx = random.randint(0, labels.shape[0] - 1)
        cur_img = images[idx]
        cur_label = labels[idx]
        in_img = test_transform(cur_img)
        in_img = in_img[np.newaxis, :]
        out = model(in_img)
        pred = torch.argmax(out).item()

        correct_counts += 1 if pred == cur_label else 0
        total_counts += 1

        print('Image idx: {}\tCorrect label: {}\tPredicted label: {}'.format(
            idx, cur_label, pred))

    accuracy = 100.0 * correct_counts / total_counts
    print('Accuracy of the network on the {0} val images: {1:.3f}%'.format(
        total_counts, accuracy))

    # Model eval
    model.eval()
    print("INFERENCE FINISHED")
