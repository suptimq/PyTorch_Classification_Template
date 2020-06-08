import os
import argparse
from pathlib import Path

import torch
import torchvision.transforms as tvtrans
import tensorboard

from helper import getTimestamp, getHostName, makeSubdir, logInfoWithDot, set_logging

from dataloader import Dataset
from solver import MNISTSolver

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--modelType',
                    default='Net',
                    help='model used for training')
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--resume_timestamp', help='timestamp to resume')
parser.add_argument('--loading_epoch',
                    type=int,
                    default=0,
                    help='xth model loaded to resume')
parser.add_argument('--total_epochs',
                    type=int,
                    default=200,
                    help='number of epochs to train for')
parser.add_argument('--outdim', type=int, default=10, help='number of classes')
parser.add_argument('--saveModel', action='store_false', help='save model')
parser.add_argument('--cuda', action='store_true', help='enable cuda')

# Optimizer
parser.add_argument('--optimType',
                    default='Adam',
                    help='optimizer used for training')
parser.add_argument('--lr',
                    type=float,
                    default=1e-4,
                    help='learning rate for optimzer')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.01,
                    help='weight decay for optimzer')

# Scheduler
parser.add_argument('--scheduler', action='store_true', help='use scheduler')
parser.add_argument('--step_size',
                    type=int,
                    default=50,
                    help='period of learning rate decay')
parser.add_argument('--gamma',
                    type=float,
                    default=0.1,
                    help='multiplicative factor of learning rate decay')

# Dataloader
parser.add_argument('--bsize', type=int, default=32, help='input batch size')
parser.add_argument('--nworker',
                    type=int,
                    default=4,
                    help='number of dataloader workers')

parser.add_argument('--manualSeed',
                    type=int,
                    default=1701,
                    help='reproduce experiemnt')
parser.add_argument('--cuda_device',
                    default="0",
                    help='ith cuda used for training')
opt = parser.parse_args()
print(opt)

torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

gpu = opt.cuda

if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device

# Basic configuration
root_path = Path(os.path.abspath(__file__))
code_root_path = root_path.parent
dataset_root_path = root_path.parents[1] / 'data'

meta_filepath = 'metadata.csv'
train_filepath = 'train.hdf5'
test_filepath = 'test.hdf5'

# Dataloader
bsize = opt.bsize
nworker = opt.nworker

# Model
current_time = getTimestamp() if not opt.resume else opt.resume_timestamp
model_type_time = opt.modelType + '_{}'.format(current_time)

model_path = code_root_path / 'models' / model_type_time
model_filename = '{0}_model_ep{1:03}'

# Logger and Writer
log_path = code_root_path / 'logs' / model_type_time
log_filename = 'log_{0}_{1}.txt'.format(
    'train', current_time)  # log_train/test_currentTime
writer_path = code_root_path / 'runs'
writer_filename = '{0}_{1}_{2}'.format(
    opt.modelType, current_time,
    getHostName())  # modelName_currentTime_hostName

# Parameters for dataset
dataset_path = {
    'root_path': dataset_root_path,
    'meta_filepath': meta_filepath,
    'train_filepath': train_filepath,
    'test_filepath': test_filepath
}

# Parameters for solver
model = {
    'model_type': opt.modelType,
    'outdim': opt.outdim,
    'resume': opt.resume,
    'loading_epoch': opt.loading_epoch,
    'total_epochs': opt.total_epochs,
    'model_path': model_path,
    'model_filename': model_filename,
    'save': opt.saveModel,
    'gpu': gpu
}

optimizer = {
    'optim_type': opt.optimType,
    'resume': opt.resume,
    'lr': opt.lr,
    'weight_decay': opt.weight_decay
}

scheduler = {
    'use': opt.scheduler,
    'step_size': opt.step_size,
    'gamma': opt.gamma
}

logging = {
    'log_path': log_path,
    'log_filename': log_filename,
    'log_level': 20,  # 20 == level (logging.INFO)
}

writer = {'writer_path': writer_path, 'writer_filename': writer_filename}

# Log config
makeSubdir(logging['log_path'])
logger = set_logging(logging['log_path'] / logging['log_filename'],
                     logging['log_level'])

# Log model, optim information
logInfoWithDot(logger, model)
logInfoWithDot(logger, optimizer)

# Preprocessing transforms: data augmentation
train_augmentation = tvtrans.Compose([
    tvtrans.ToPILImage(),
    tvtrans.RandomHorizontalFlip(p=0.5),
    tvtrans.RandomVerticalFlip(p=0.5),
    tvtrans.ToTensor(),
    tvtrans.Normalize((0.5, ), (0.5, ))
])

test_transform = tvtrans.Compose([
    tvtrans.ToPILImage(),
    tvtrans.ToTensor(),
    tvtrans.Normalize((0.5, ), (0.5, ))
])

# Get MNIST dataset
train_ds = Dataset(dataset_path,
                   train=True,
                   transform=train_augmentation)
train_dl = torch.utils.data.DataLoader(train_ds,
                                       batch_size=bsize,
                                       shuffle=True)

test_ds = Dataset(dataset_path,
                  train=False,
                  transform=test_transform)
test_dl = torch.utils.data.DataLoader(test_ds,
                                      batch_size=bsize,
                                      shuffle=False)

dataloader = {'train': train_dl, 'valid': test_dl}

solver = MNISTSolver(model, dataloader, optimizer, scheduler, logger, writer)
solver.forward()
