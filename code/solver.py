### Solver (train and test)
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from model import Net
from helper import makeSubdir, logInfoWithDot, timeSince


class Solver():
    def __init__(self, model, dataloader, optimizer, scheduler, logger, writer):
        """
        Args:
            model:         Model params
            dataloader:    Dataloader params
            optimizer:     Optimizer params
            scheduler:     Scheduler params
            logger:        Logger params
            writer:        Tensorboard writer params
        """
        self.model = self.init_model(model)
        self.loading_epoch = model['loading_epoch']
        self.total_epochs = model['total_epochs']
        self.model_path = model['model_path']
        self.model_filename = model['model_filename']
        self.save = model['save']
        self.model_name = model['model_type']
        self.model_fullpath = str(self.model_path / self.model_filename)

        # Logger
        self.logger = logger
        # Writer: Default path is runs/CURRENT_DATETIME_HOSTNAME
        writer_fullname = writer['writer_path'] / writer['writer_filename']
        self.writer = SummaryWriter(log_dir=writer_fullname)

        if model['gpu'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)
            logInfoWithDot(self.logger, "USING GPU")
        else:
            self.device = torch.device('cpu')
            logInfoWithDot(self.logger, "USING CPU")

        if model['resume']:
            load_model_path = self.model_fullpath.format(
                self.model_name, self.loading_epoch)
            checkpoint = torch.load(load_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Log information
            logInfoWithDot(self.logger, "LOADING MODEL FINISHED")

        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self.init_optimizer(optimizer)
        if optimizer['resume']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Log information
            logInfoWithDot(self.logger, "LOADING OPTIMIZER FINISHED")

        # Scheduler
        self.scheduler = None
        if scheduler['use']:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler['step_size'],
                gamma=scheduler['gamma'])

        # Dataloader
        self.trainloader = dataloader['train']
        self.validloader = dataloader['valid']

        # Evaluation
        self.train_loss = 0
        self.correct_counts = 0
        self.total_counts = 0

        # Timer
        self.start_time = time.time()

    def init_model(self, model):
        m = None
        outdim = model['outdim']
        m = Net(outdim)

        assert m != None, 'Model Not Initialized'
        return m

    def init_optimizer(self, optimizer):
        opt = None
        lr = optimizer['lr']
        weight_decay = optimizer['weight_decay']
        if optimizer['optim_type'] == 'Adam':
            opt = optim.Adam(self.model.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
        elif optimizer['optim_type'] == 'Adadelta':
            opt = optim.Adadelta(self.model.parameters(),
                                 weight_decay=weight_decay)
        else:
            opt = optim.RMSprop(self.model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)

        assert opt != None, 'Optimizer Not Initialized'
        return opt

    def count_correct(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).sum().item()

    def train_one_epoch(self, ep):
        pass

    def test_one_epoch(self, ep):
        pass

    def forward(self):
        start_epoch = self.loading_epoch + 1
        total_epochs = start_epoch + self.total_epochs
        for epoch in range(start_epoch, total_epochs):
            self.train_one_epoch(epoch)
            self.test_one_epoch(epoch)

            # Update learning rate after every specified times iteration
            if self.scheduler:
                self.scheduler.step()

            if self.save:
                makeSubdir(self.model_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.model_fullpath.format(self.model_name, epoch))

                # Log information
                logInfoWithDot(
                    self.logger, "SAVED MODEL: {}".format(
                        self.model_fullpath.format(self.model_name, epoch)))

        # self.writer.add_graph(self.model)
        self.writer.close()
        logInfoWithDot(
            self.logger, "TRAINING FINISHED, TIME USAGE: {} secs".format(
                timeSince(self.start_time)))


class MnistSolver(Solver):
    def train_one_epoch(self, ep, log_interval=50):
        self.model.train()

        lr = self.optimizer.param_groups[0]['lr']

        for i, (images, labels) in enumerate(self.trainloader, 0):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Inference
            preds = self.model(images)
            self.correct_counts += self.count_correct(preds, labels)
            self.total_counts += len(labels)
            # Loss
            loss = self.criterion(preds, labels)
            self.train_loss += loss.item()
            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % log_interval == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLearning Rate: {}\tLoss: {:.6f}\tTime Usage:{:.8}'
                    .format(ep, i * len(images), len(self.trainloader.dataset),
                            100. * i / len(self.trainloader), lr, loss.data,
                            timeSince(self.start_time)))

            if i == len(self.trainloader) - 1:
                accuracy = 100.0 * self.correct_counts / self.total_counts
                self.logger.info(
                    'Loss of the network on the {0} train images: {1:.6f}'.
                    format(self.total_counts, self.train_loss))
                self.logger.info(
                    'Accuracy of the network on the {0} train images: {1:.3f}%'
                    .format(self.total_counts, accuracy))

                # Write to Tensorboard file
                self.writer.add_scalar('Loss/train', self.train_loss, ep)
                self.writer.add_scalar('Accuracy/train', accuracy, ep)

    def test_one_epoch(self, ep):
        self.model.eval()

        eval_loss = 0
        correct_counts = 0
        total_counts = 0
        for _, (images, labels) in enumerate(self.validloader, 0):
            images = images.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(images)
            correct_counts += self.count_correct(preds, labels)
            total_counts += len(labels)
            with torch.no_grad():
                loss = self.criterion(preds, labels)
                eval_loss += loss.item()

        accuracy = 100.0 * correct_counts / total_counts
        self.logger.info(
            'Loss of the network on the {0} val images: {1:.6f}'.format(
                total_counts, eval_loss))
        self.logger.info(
            'Accuracy of the network on the {0} val images: {1:.3f}%'.format(
                total_counts, accuracy))

        # Write to Tensorboard file
        self.writer.add_scalar('Loss/val', eval_loss, ep)
        self.writer.add_scalar('Accuracy/val', accuracy, ep)
