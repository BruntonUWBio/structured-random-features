import numpy as np
import torch
import torch.nn as nn

from src.models.init_weights import sensilla_init, V1_init, classical_init, V1_weights

class V1_mnist_RFNet(nn.Module):
    """
    Random Feature network to classify MNIST images. The first layer is initialized from GP
    with covariance inspired by V1. The layers are convolutional layers with kernels covering
     the entire dataset.
    """
    def __init__(self, hidden_dim, size, spatial_freq, center=None, scale=1, bias=False, seed=None):
        super(V1_mnist_RFNet, self).__init__()
        self.v1_layer = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=28) 
        self.clf = nn.Conv2d(in_channels=hidden_dim, out_channels=10, kernel_size=1)
        self.relu = nn.ReLU()
        
        # initialize the first layer
        V1_init(self.v1_layer, size, spatial_freq, center, scale, bias, seed)
        self.v1_layer.weight.requires_grad = False
        
    def forward(self, x):
        h = self.relu(self.v1_layer(x))
        beta = self.clf(h)
        return beta.squeeze()


class sensilla_RFNet(nn.Module):
    """
    Random Feature network to classify time-series. The first layer is initialized from GP
    with covariance inspired by mechanosensory sensilla.
    """
    def __init__(self, input_dim, hidden_dim, 
                 lowcut, highcut, decay_coef=np.inf, scale=1, bias=False, seed=None):
        super(sensilla_RFNet, self).__init__()
        self.sensilla_layer = nn.Linear(in_features=input_dim, out_features=hidden_dim) 
        self.clf = nn.Linear(in_features=hidden_dim, out_features=2)
        self.relu = nn.ReLU()
        
        # initialize the first layer
        sensilla_init(self.sensilla_layer, lowcut, highcut, decay_coef, scale, bias, seed)
        self.sensilla_layer.weight.requires_grad = False
        
    def forward(self, x):
        h = self.relu(self.sensilla_layer(x))
        beta = self.clf(h)
        return beta.squeeze()


class classical_RFNet(nn.Module):
    """
    Random Feature network to classify time-series or MNIST digits. The first layer is initialized from GP
    with diagonal covariance.
    """
    def __init__(self, input_dim, hidden_dim, scale=1, bias=False, seed=None):
        super(classical_RFNet, self).__init__()
        if type(input_dim) is int: ## for time-series
            self.RF_layer = nn.Linear(in_features=input_dim, out_features=hidden_dim) 
            self.clf = nn.Linear(in_features=hidden_dim, out_features=2)
        elif type(input_dim) is tuple: ## for MNIST
            self.RF_layer = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=28)
            self.clf = nn.Conv2d(in_channels=hidden_dim, out_channels=10, kernel_size=1)
        self.relu = nn.ReLU()
        
        # initialize the first layer
        classical_init(self.RF_layer, scale, bias, seed)
        self.RF_layer.weight.requires_grad = False
        
    def forward(self, x):
        h = self.relu(self.RF_layer(x))
        beta = self.clf(h)
        return beta.squeeze()