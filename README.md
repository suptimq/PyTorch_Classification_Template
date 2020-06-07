# Template for PyTorch Classification Task

A well-defined template to do training and inference for classification task when using PyTorch framework

### Requirements

- [Python 3.6](https://www.python.org/)
- [PyTorch >= 1.4](https://pytorch.org/)
- [TorchVision](https://pypi.org/project/torchvision/)

### Structure

The basic filesystem structure is like what the following tree displays. Essentially, we have two folders **code** and **data** to store scripts and dataset, respectively.

```console
PyTorch/
├── code
│   ├── models
│   ├── runs
│   ├── logs
│   ├── dataloader.py
│   ├── helper.py
│   ├── run.py
│   ├── solver.py
│   ├── model.py
│   └── run.sh
└── data
    ├── test.hdf5
    └── train.hdf5
```

#### Folders

Within the **code** folder, there are three subfolders **models**, **logs**, and **runs**. They will be generated after executing the code

```console
- models: Save trained model parameter files
- logs: Save training and inference logs
- runs: Save training and validation figures
```

Within the **data** foler, there are one file storing the metadata for the raw data and another two dataset files saved as the HDF5 format

#### Files

```console
- run.py: Most of training or inference related variables are declared in this file
- run.sh: The program entry script where configurations can be customized
- helper: Miscellaneous helper functions
```

### Data Preparation

For the specific example, we use the [MNIST](https://www.kaggle.com/benedictwilkinsai/mnist-hd5f) dataset from Kaggle. The data are stored by HDF5 so they are simply imported using Python *h5py* library in the **dataloader.py** file.

### Running

The training and validation process are executed in the **solver.py** file while the entry point is the **run.py** file, where the logger, the dataloader, and the solver are instanciated


#### Training

> **_NOTE:_**  The modelType argument stated below is not yet available since the state-of-the-art architectures support only 3-channel image, whereas the MNIST images have only 1 channel. Therefore, we simply define a shallow CNN in the **model.py** file

Training a model from scratch is quite simple, just specify the above arguments such as pretrained model type, optimizer type, and other hpyerparameters in the **run.sh** file

```Python
#! run.sh

python /{your_path}/run.py          \
        --modelType GoogleNet       \
        --loading_epoch 0           \
        --total_epochs 200          \
        --cuda                      \
        --optimType Adam            \
        --bsize 32                  \
        --nworker 4                 \
        --cuda_device 0
```

Resume training models is also not too much difficult. Since the logs and models are organized in the following structure, where each time we run experiments, these three folders will be made with the almost same filename. 

```console
PyTorch/
└── code
    ├── ...
    ├── logs
    │   ├── DenseNet_Jun04_21-09-01
    │   ├── GoogleNet_Jun05_21-31-54
    │   ├── ResNet_Jun04_21-04-42
    │   └── SqueezeNet_Jun04_18-18-46
    ├── models
    │   ├── DenseNet_Jun04_21-09-01
    │   ├── GoogleNet_Jun05_21-31-54
    │   ├── ResNet_Jun04_21-04-42
    │   └── SqueezeNet_Jun04_18-18-46
    └── runs
        ├── DenseNet_Jun04_21-09-01_cornell-rbtg
        ├── GoogleNet_Jun05_21-31-54_cornell-rbtg
        ├── ResNet_Jun04_21-04-42_cornell-rbtg
        └── SqueezeNet_Jun04_18-18-46_cornell-rbtg
```

Therefore, as we would like to resume training from certain checkpoint, we only need to configure the **run.sh** file like this:

```Python
#! run.sh

python /{your_path}/run.py          \
        --modelType GoogleNet       \
        ###
        --resume                    \
        --resume_timestamp Jun05_21-31-54    --loading_epoch {epoch resume from}       \
        ###
        --total_epochs 200          \
        --cuda                      \
        --optimType Adam            \
        --bsize 32                  \
        --nworker 4                 \
        --cuda_device 0
```

### Inference

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
