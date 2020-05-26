import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# from pdb import set_trace

# Basic Neural Network Module


class ConvModule(nn.Module):
    def __init__(self, in_, out_, kernel_size=3, stride=1, padding=1):
        super(ConvModule, self).__init__()
        """
        Args:
            in_:    input feature dimensions
            out_:   output feature dimensions
            With max pooling size fixed (2)
        """
        # Input Shape: (N, C_in, L)
        self.conv = nn.Conv1d(in_, out_, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
        """
        Source: https://github.com/dannysdeng/selu
        """
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, x):
        temp1 = self.scale * F.relu(x)
        temp2 = self.scale * self.alpha * (F.elu(-1 * F.relu(-1 * x)))
        return temp1 + temp2


class alpha_drop(nn.Module):
    def __init__(self, p=0.05, alpha=-1.7580993408473766, fixedPointMean=0, fixedPointVar=1):
        super(alpha_drop, self).__init__()
        """
        Source: https://github.com/dannysdeng/selu
        """
        keep_prob = 1 - p
        self.a = np.sqrt(fixedPointVar / (keep_prob * ((1-keep_prob)
                                                       * pow(alpha-fixedPointMean, 2) + fixedPointVar)))
        self.b = fixedPointMean - self.a * \
            (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        self.alpha = alpha
        self.keep_prob = 1 - p
        self.drop_prob = p

    def forward(self, x):
        if self.keep_prob == 1 or not self.training:
            # print("testing mode, direct return")
            return x
        else:
            random_tensor = self.keep_prob + torch.rand(x.size())

            binary_tensor = Variable(torch.floor(random_tensor))

            if torch.cuda.is_available():
                binary_tensor = binary_tensor.cuda()

            x = x.mul(binary_tensor)
            ret = x + self.alpha * (1-binary_tensor)
            ret.mul_(self.a).add_(self.b)
            return ret

# CNN Model for signal classification
### Source: "Over the Air Deep Learning Based Radio Signal Classification"


class SignalCNN(nn.Module):
    def __init__(self, output_, hnrn=128, rnn=False, layer_num=7):
        super(SignalCNN, self).__init__()
        self.modules = []
        self.layer_num = layer_num
        self.output_ = output_

        self.addLayer()
        # CNN output shape: (bsize, channel, sequence length)
        self.conv = nn.Sequential(*self.modules)

        self.rnn = None
        if rnn:
            # Input shape: (sequence length, bsize, channel)
            self.rnn = nn.LSTM(64, 64, num_layers=3,
                               dropout=0.5, bidirectional=False)

        self.fc = nn.Sequential(
            nn.Linear(64 * 8, hnrn),
            selu(),
            nn.Dropout(p=0.3),
            nn.Linear(hnrn, hnrn),
            selu(),
            nn.Dropout(p=0.3)
        )

        self.fc_last = nn.Linear(hnrn, output_)

    def addLayer(self):
        for i in range(self.layer_num):
            if (i == 0):
                module = ConvModule(2, 64)
            else:
                module = ConvModule(64, 64)
            self.modules.append(module)

    def forward(self, x):
        x = self.conv(x)
        if self.rnn:
            rnn_in = x.transpose(0, 1).transpose(0, 2)
            # Shape: (sequence length, bsize, channel)
            rnn_out, _ = self.rnn(rnn_in)
            x = rnn_out.transpose(0, 1)

        fc_in = x.reshape(x.size(0), -1)
        fc_out = self.fc(fc_in)
        out = self.fc_last(fc_out)
        return out

    def __repr__(self):
        return 'cnn'


# Resnet Model

class ResnetStack(nn.Module):
    def __init__(self, in_, out_, res_num=2, kernel_size=3, stride=1, padding=1):
        super(ResnetStack, self).__init__()
        """
        Args:
            in_:    input feature dimensions
            out_:   output feature dimensions
            With max pooling size fixed (2)
        """
        # Input Shape: (N, C_in, L)
        # 1 * 1 conv layer
        self.conv = nn.Conv1d(in_, out_, kernel_size=1, stride=1, padding=0)
        # Residual unit
        self.res_module = nn.Sequential(
            nn.Conv1d(out_, out_, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(out_, out_, kernel_size=1, stride=1, padding=0)
        )
        self.pool = nn.MaxPool1d(2)

        self.res_num = res_num

    def forward(self, x):
        x = self.conv(x)
        for _ in range(self.res_num):
            x_shortcut = x
            x = self.res_module(x) + x_shortcut
        x = self.pool(x)

        return x


class SignalResnet(nn.Module):
    def __init__(self, output_, hnrn=128, rnn=False, layer_num=6):
        super(SignalResnet, self).__init__()
        self.modules = []
        self.layer_num = layer_num
        self.output_ = output_

        self.add_layer()
        # CNN output shape: (bsize, channel, sequence length)
        self.conv = nn.Sequential(*self.modules)

        self.fc = nn.Sequential(
            nn.Linear(32 * 16, hnrn),
            selu(),
            alpha_drop(p=0.3),
            nn.Linear(hnrn, hnrn),
            selu(),
            alpha_drop(p=0.3)
        )

        self.fc_last = nn.Linear(hnrn, output_)

    def add_layer(self):
        for i in range(self.layer_num):
            if i == 0:
                module = ResnetStack(2, 32)
            else:
                module = ResnetStack(32, 32)
            self.modules.append(module)

    def forward(self, x):
        x = self.conv(x)
        fc_in = x.reshape(x.size(0), -1)
        fc_out = self.fc(fc_in)
        out = self.fc_last(fc_out)
        return out

    def __repr__(self):
        return 'resnet'


# A Shallow CNN Model


class FeedFrwdNet(nn.Module):
    def __init__(self, ip_dim, hid_nrn, op_dim):
        self.ip_dim = ip_dim
        super(FeedFrwdNet, self).__init__()
        self.lin1 = nn.Linear(ip_dim, hid_nrn)  # First hidden layer
        self.lin2 = nn.Linear(hid_nrn, op_dim)  # Output layer with 10 neurons

    def forward(self, input):
        x = input.view(-1, self.ip_dim)  # Sort of a numpy reshape function
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
