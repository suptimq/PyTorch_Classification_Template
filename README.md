# Template for PyTorch Classification Task

A well-defined template to do training and inference for classification task when using PyTorch framework

## Structure

### Folders

- datasets: save training, testing, and validation datasets
- models: save trained model parameter files
- logs: save training and inference logs
- figures: save training and validation figures

### Files

- config: Most of training or inference related variables are declared in this file
- helper: Miscellaneous helper functions

## Data Preparation

For the specified code in this repository, the data are stored by HDF5 so they are simply imported using Python *h5py* library in the **dataloader.py** file. However, it is easy to customize your own dataset with only a little bit change in code

```Python
# Convert the original labels to the numerical format if necessary
classes = ['A', 'B', ..., 'K']
label_dict, label_dict_reverse = labelConverter(classes)
```

## Running

The training and validation process are executed in the **solver.py** file while the entry point is the **run.py** file, where the logger, the dataloader, and the solver are instanciated

```Python
# Split dataset into training set and validation set
split_size = 0.1
X_train, X_val, y_train, y_val = train_test_split(train_data_total,
                                                  train_labels_total,
                                                  test_size=split_size)

# Define the solver
solver = SignalSolver(model, dataloader, optimizer, scheduler, logger)
```

## Inference

Inference can be easily done by running **inference.py** file as follows:

```console
Python inference.py
```

It is going to generate a CSV file indicating the label for each of the testing data

```Python
# Model eval
model.eval()
results = inference(model, label_dict_reverse, test_data, saved_path, logger)
```