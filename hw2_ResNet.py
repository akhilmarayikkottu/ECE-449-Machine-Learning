#import matplotlib.pyplot as plt
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim



class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, num_channels, kernel_size =3 , stride=1, padding=1, bias=False)
        self.bn1   = torch.nn.BatchNorm2d(num_channels)
        self.relu  = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(num_channels, num_channels, kernel_size =3 , stride=1, padding=1, bias=False)
        self.bn2   = torch.nn.BatchNorm2d(num_channels)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out+residual
        out = self.relu(out)
        return(out)
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        pass


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, num_channels, kernel_size =3 , stride=2, padding=1, bias=False)
        self.bn1   = torch.nn.BatchNorm2d(num_channels)
        self.relu  = torch.nn.ReLU(inplace=True)
        self.maxp  = torch.nn.MaxPool2d(kernel_size=2)
        self.blk   = Block(num_channels)
        self.Avg2d = torch.nn.AdaptiveAvgPool2d(1)
        self.line  = torch.nn.Linear(num_channels, num_classes)

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        residual = x
       # print("size of input", x.size())
        out = self.conv1(x)
       # print("size of out conv1", out.size())
        out = self.bn1(out)
       # print("size of out bn1", out.size())
        out = self.relu(out)
       # print("size of out relu", out.size())
        out = self.maxp(out)
       # print("size of out maxp", out.size())
        out = self.blk(out)
       # print("size of out blk", out.size())
        out = self.Avg2d(out)
       # print("size of out Avg2d", out.size())
        out = nn.Flatten()(out)
       # print("flatten " , out.size())
        out = self.line(out)
       # print("size of out linear", out.size())

        return(out) 

        pass
