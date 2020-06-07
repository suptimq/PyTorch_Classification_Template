import h5py
import torch
import pandas as pd
import numpy as np

from logger import set_logging
from config import *
from helper import labelConverter, logInfoWithDot, makeSubdir, getTimestamp
from model import SignalCNN, SignalResnet


def inference(model, label_dict_reverse, data, path, logger):
    # Doing inference and generate CSV file
    """
    Args:
        model:
        label_dict_reverse:
        data:
        path:
    """

    results = []
    for d in data:
        d = d[np.newaxis, ...]
        input = torch.from_numpy(d).transpose(1, 2).float()
        ouput = model(input)
        _, predicted = torch.max(ouput.data, axis=1)
        results.append(predicted)

    # Generatet CSV file
    real_labels = [label_dict_reverse[out.numpy()[0]] for out in results]
    df_real_labels = pd.DataFrame(real_labels, columns=['Category'])
    df_real_labels['Id'] = df_real_labels.index
    df_real_labels.to_csv(path_or_buf=path, index=False)

    logInfoWithDot(logger, 'GENERATED OUTPUT CSV FILE TO: {}'.format(path))

    return results


dataset_filename = './datasets/data.hdf5'
result_path = "./results/{0}_out_{1:03}.csv"
makeSubdir(result_path)

# Log config
makeSubdir(logging['log_path'])
logger = set_logging(
    logging['log_path'] + logging['log_filename'].format('inference', getTimestamp()), logging['log_level'])


# Import data from HDF5
f = h5py.File(dataset_filename, 'r')
test_data = f['test']

# Load model
# model = SignalCNN(output_dim)
model = SignalResnet(outdim)
model_name = str(model)
model_path = model['model_path'].format(model_name, inference_loading)
saved_path = result_path.format(model_name, inference_loading)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

logInfoWithDot(logger, "LOADING MODEL FINISHED")

# Convert the original labels to the numerical format
classes = [
    'FM', 'OQPSK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', 'AM-DSB-SC',
    'QPSK', 'OOK'
]
label_dict, label_dict_reverse = labelConverter(classes)

logInfoWithDot(logger, "INFERENCE START")


# Model eval
model.eval()
results = inference(model, label_dict_reverse, test_data, saved_path, logger)

logInfoWithDot(logger, "INFERENCE FINISHED")
