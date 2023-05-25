"""
Implementation of original AlexNet architecture from the paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky et al.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

print(torch.cuda.is_available())

# hyperparameters based on the original paper
BATCH_SIZE = 128
EPOCHS = 90
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
IMAGE_SIZE = 227
NUM_CLASSES = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class AlexNet(nn.Module):

    def __init__(self, image_size = 227, num_classes = 1000):
        
        """
        Args:
            image_size: size of input image
            num_classes: number of classes in the dataset
        """
        
        super.__init__()

        self.image_size = image_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)
