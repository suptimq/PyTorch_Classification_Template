import os
import time
import math
import socket
from datetime import datetime
import logging


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
