import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        
        ## Initialize the block with a call to super and make your conv and batchnorm layers.
        super(ResNetBlock, self).__init__()
        # TODO: Initialize conv and batch norm layers with the correct parameters
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        ## Use some conditional logic when defining your shortcut layer
        ## For a no-op layer, consider creating an empty nn.Sequential()
        self.shortcut = None  # ???
        # TODO: Code here to initialize the shortcut layer
        if stride != 1 or out_channels * 4 != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels * 4),
            )
        else:
            self.shortcut = nn.Sequential()
        ## END YOUR CODE

    def forward(self, x):
        """
        Compute a forward pass of this batch of data on this residual block.

        x: batch of images of shape (batch_size, num_channels, width, height)
        returns: result of passing x through this block
        """
        ## YOUR CODE HERE
        ## TODO: Call the first convolution, batchnorm, and activation
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(x)))
        ## TODO: Call the second convolution and batchnorm
        out = self.bn3(self.conv3(out))
        ## TODO: Also call the shortcut layer on the original input
        out += self.shortcut(x)
        ## TODO: Sum the result of the shortcut and the result of the second batchnorm
        ## and apply your activation
        out = F.relu(out)
        return out
        ## END YOUR CODE


class ResNet50(nn.Module):
    def __init__(self, num_classes=200):
        # Read the following, and uncomment it when you understand it, no need to add more code
        self.num_classes = num_classes
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_block(out_channels=64, stride=1, blocks=3)
        self.layer2 = self.make_block(out_channels=128, stride=2, blocks=4)
        self.layer3 = self.make_block(out_channels=256, stride=2, blocks=6)
        self.layer4 = self.make_block(out_channels=512, stride=2, blocks=3)
        self.linear = nn.Linear(2048, num_classes)

    def make_block(self, out_channels, stride, blocks):
        # Read the following, and uncomment it when you understand it, no need to add more code
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * 4
        for _ in range(blocks - 1):
            layers.append(ResNetBlock(self.in_channels, out_channels, 1))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        # Read the following, and uncomment it when you understand it, no need to add more code
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
