import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import time

from config import *
from helper import *
from logger import set_logging

from dataloader import train_data_total, train_labels_total, SignalDataset
from solver import SignalSolver

np.random.seed(2020)


# Log config
makeSubdir(logging['log_path'])
logger = set_logging(
    logging['log_path'] + logging['log_filename'].format('train', getTimestamp()), logging['log_level'])

# Split dataset into training set and validation set
split_size = 0.1
X_train, X_val, y_train, y_val = train_test_split(train_data_total,
                                                  train_labels_total,
                                                  test_size=split_size)

logInfoWithDot(logger, "DATA PREPARATION FINISHED")

# Get Signal dataset
trainset = SignalDataset(X_train, y_train)
trainloader = DataLoader(trainset, batch_size=bsize, shuffle=True)

valset = SignalDataset(X_val, y_val)
valloader = DataLoader(valset, batch_size=bsize, shuffle=False)

dataloader = {
    'train': trainloader,
    'valid': valloader
}

solver = SignalSolver(model, dataloader, optimizer, scheduler, logger)

# Timer
start_time = time.time()

train_loss, tst_acc = solver.run()

logInfoWithDot(logger, "TRAINING FINISHED, TIME USAGE: {}".format(
    time.time() - start_time))

# Plot
plot(model['total_epochs'], train_loss, tst_acc)
