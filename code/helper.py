import os
import time
import math
import socket
from datetime import datetime
import logging

import torch.optim as optim

from model import Net


def set_logging(log_file, log_level=logging.DEBUG):
    """
    Logging to console and log file simultaneously.
    """
    log_format = '[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s'
    formatter = logging.Formatter(log_format)
    logging.basicConfig(level=log_level, format=log_format, filename=log_file)
    # Console Log Handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(log_level)
    logging.getLogger().addHandler(console)
    return logging.getLogger()


def getTimestamp(f='%b%d_%H-%M-%S'):
    return datetime.now().strftime(f)


def makeSubdir(dirname):
    # Recursively make directories
    os.makedirs(dirname, exist_ok=True)


def getHostName():
    return socket.gethostname()


def logInfoWithDot(logger, text):
    logger.info(text)
    logger.info('--------------------------------------------')


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))


def init_model(model):
    m = None
    outdim = model['outdim']
    m = Net(outdim)

    assert m != None, 'Model Not Initialized'
    return m


def init_optimizer(optimizer, model):
    opt = None
    lr = optimizer['lr']
    weight_decay = optimizer['weight_decay']
    parameters = model.parameters()
    if optimizer['optim_type'] == 'Adam':
        opt = optim.Adam(parameters,
                         lr=lr,
                         weight_decay=weight_decay)
    elif optimizer['optim_type'] == 'Adadelta':
        opt = optim.Adadelta(parameters,
                             weight_decay=weight_decay)
    else:
        opt = optim.RMSprop(parameters,
                            lr=lr,
                            weight_decay=weight_decay)

    assert opt != None, 'Optimizer Not Initialized'
    return opt
