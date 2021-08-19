import numpy as np
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Any

from src.models.init_weights import sensilla_init, V1_init, classical_init

class V1_mnist_RFNet(nn.Module):
    """
    Random Feature network to classify MNIST images. The first layer is initialized from GP
    with covariance inspired by V1.
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


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = False, structured: bool = False,
            progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        structured (bool): If True, initialize with structured weights, 
                           otherwise use classical (Gaussian) init
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    
    def init_weights(m):
        '''
        Only update the Conv2d layers
        '''
        if isinstance(m, nn.Conv2d):
            C_out, C_in, xdim, ydim = m.weight.shape
            scale = 1 / (C_in * xdim ** 2 * ydim ** 2)
            if structured:
                # idea: could make size & freq scale with conv dimensions
                V1_init(m, size=5, spatial_freq=2, bias=True, scale=scale)
            else:
                classical_init(m, bias=True, scale=scale)
    
    model = AlexNet(**kwargs)
    model.apply(init_weights)
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth', progress=progress)
        model.load_state_dict(state_dict)
    return model
