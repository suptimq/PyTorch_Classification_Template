import torch.optim as optim

from model import SignalCNN, SignalResnet

# Basic configuration
bsize = 64
indim = 2 * 1024
outdim = 10
hndim = 128
resume = False
loading_epoch = 0
total_epochs = 2
model_path = './models/{0}_model_ep{1:03}'
save = True

lr = 1e-3
step_size = 20
gamma = 0.1

inference_loading = 10

# Parameters for solver
model = {
    'model_type': SignalResnet,
    'outdim': outdim,
    'hndim': hndim,
    'rnn': False,
    'resume': resume,
    'loading_epoch': loading_epoch,
    'total_epochs': total_epochs,
    'model_path': model_path,
    'save': save
}

optimizer = {
    'optim_type': optim.Adam,
    'resume': resume,
    'lr': lr,
}

scheduler = {
    'use': False,
    'step_size': step_size,
    'gamma': gamma
}

logging = {
    'log_path': './logs/',
    'log_filename': 'log_{}_{}',   # train or test
    'log_level': 20,               # logging.INFO
}
