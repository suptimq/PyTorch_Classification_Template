import matplotlib.pyplot as plt
from time import gmtime, strftime
import os


def labelConverter(classes):
    # Convert text labels to numerical labels

    label_dict = {}
    label_dict_reverse = {}
    for idx, cla in enumerate(classes):
        label_dict[cla] = idx
        label_dict_reverse[idx] = cla

    return label_dict, label_dict_reverse


def getTimestamp(f='%Y-%m-%d-%H-%M-%S'):
    return strftime(f, gmtime())


def makeSubdir(fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def logInfoWithDot(logger, text):
    logger.info('--------------------------------------------')
    logger.info(text)
    logger.info('--------------------------------------------')


def plot(epochs, train_loss, tst_acc, fname='./figures/out_{}.png'):
    makeSubdir(fname)

    # Plot training lossa and validation accuracy

    plt.figure(1)
    plt.plot(range(epochs), train_loss, label='train loss')
    plt.legend()
    # Save figures
    plt.savefig(fname.format('train_loss_'+getTimestamp()))

    plt.figure(2)
    plt.plot(range(epochs), tst_acc, label='test acc')
    plt.legend()
    # Save figures
    plt.savefig(fname.format('test_acc_'+getTimestamp()))
