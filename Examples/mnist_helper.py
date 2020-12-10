import torch
from torchvision import datasets, transforms
import numpy as np

import sys
sys.path.append('../OTDD/')
from otdd_pytorch import TensorDataset

def get_mnist_data():
    mnist_transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

    mnist_train = datasets.MNIST('../data', train=True, download=True,
                       transform=mnist_transform)
    
    mnist_feats = []
    mnist_labels = []

    for i in range(len(mnist_train)):
        feats, labels = mnist_train[i]
        mnist_feats.append(feats)
        mnist_labels.append(labels)

    mnist_feats = torch.cat(mnist_feats).view(len(mnist_train), -1)
    mnist_labels = torch.tensor(mnist_labels)

    mnist_data = TensorDataset(mnist_feats, mnist_labels)
    
    return mnist_data

def get_usps_data():
    
    usps_transform = transforms.Compose([
                                transforms.Resize((28,28), interpolation=2),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

    usps_train = datasets.USPS('../data', train=True, download=True,
                       transform=usps_transform)

    usps_feats = []
    usps_labels = []

    for i in range(len(usps_train)):
        feats, labels = usps_train[i]
        usps_feats.append(feats)
        usps_labels.append(labels)

    usps_feats = torch.cat(usps_feats).view(len(usps_train), -1)
    usps_labels = torch.tensor(usps_labels)

    usps_data = TensorDataset(usps_feats, usps_labels)
    
    return usps_data

