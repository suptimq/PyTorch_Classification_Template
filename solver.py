### Solver (train and test)
import torch
import torch.nn as nn
import torch.optim as optim
import os

from logger import set_logging
from helper import makeSubdir, getTimestamp, logInfoWithDot

from pdb import set_trace


class SignalSolver():
    def __init__(self, model, dataloader, optimizer, scheduler, logger):
        """
        Args:
            model:         Model params
            dataloader:    Dataloader params
            optimizer:     Optimizer params
            scheduler:     Scheduler params
            logger:        Logger params
        """
        self.model = model['model_type'](
            model['outdim'], model['hndim'], model['rnn'])
        self.loading_epoch = model['loading_epoch']
        self.total_epochs = model['total_epochs']
        self.model_path = model['model_path']
        self.save = model['save']
        self.model_name = str(self.model)

        self.logger = logger

        if model['resume']:
            load_model_path = self.model_path.format(self.loading_epoch)
            checkpoint = torch.load(load_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Log information
            logInfoWithDot(self.logger, "LOADING MODEL FINISHED")

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = optimizer['optim_type'](
            self.model.parameters(), lr=optimizer['lr'])
        if optimizer['resume']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Log
            logInfoWithDot(self.logger, "LOADING OPTIMIZER FINISHED")

        self.scheduler = None
        if scheduler['use']:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler['step_size'],
                gamma=scheduler['gamma'])

        self.trainloader = dataloader['train']
        self.validloader = dataloader['valid']

        self.train_loss = []
        self.test_acc = []

    def train(self, epoch, log_interval=100):
        self.model.train()

        lr = self.optimizer.param_groups[0]['lr']

        # set_trace()

        for batch_idx, sample in enumerate(self.trainloader, 0):
            inputs, labels = sample.values()
            labels = labels.long()

            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss_v = self.loss(output, labels)

            loss_v.backward()
            self.optimizer.step()

            if batch_idx % log_interval == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLearning Rate: {}\tLoss: {:.6f}'
                    .format(epoch, batch_idx * len(inputs),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader),
                            lr,
                            loss_v.data))

            if batch_idx == len(self.trainloader) - 1:
                self.train_loss.append(loss_v.data)

    def valid(self):
        correct = 0
        total = 0
        for sample in self.validloader:
            signals = sample['signal']
            labels = sample['label'].long()

            outputs = self.model(signals)
            _, predicted = torch.max(outputs.data, axis=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        self.logger.info('Accuracy of the network on the %d test images: %f %%' %
                         (total, 100.0 * correct / total))
        self.test_acc.append(100.0 * correct / total)

    def run(self):
        start_epoch = self.loading_epoch + 1
        total_epochs = start_epoch + self.total_epochs
        for epoch in range(start_epoch, total_epochs):
            self.train(epoch)
            self.valid()

            if self.save:
                makeSubdir(self.model_path)
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, self.model_path.format(self.model_name, epoch))

                # Log information
                logInfoWithDot(self.logger, "SAVED MODEL: {}".format(
                    self.model_path.format(self.model_name, epoch)))

            # Update learning rate after every specified times iteration
            if self.scheduler:
                self.scheduler.step()

        return self.train_loss, self.test_acc
