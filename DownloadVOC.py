import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import os
from torchsummary import summary

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
DOWNLOAD_ImageNet = True

# Mnist digits dataset
if not (os.path.exists('./VOCDetection/')) or not os.listdir('./VOCDetection/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_ImageNet = True

train_data = torchvision.datasets.VOCDetection(
    root='./VOCDetection/',
    # train=False,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    # download=DOWNLOAD_MNIST,
    download=DOWNLOAD_ImageNet,
)
