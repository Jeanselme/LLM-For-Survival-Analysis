from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch

def create_nn(inputdim, layers, layer_unit = nn.Linear, activation = 'ReLU'):
    """
        Create a simple multi layer perceptron
    """
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(layer_unit(prevdim, hidden, bias = True))
        modules.append(act)
        prevdim = hidden

    return modules

class DeepHitTorch(nn.Module):
    """
        DeepHit torch model
    """

    def __init__(self, inputdim, layers, splits):
        """
        Args:
            embeddim (int): Input dimension (hidden state)
            splits (list int): Splits for the outcome data.
            survival_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(DeepHitTorch, self).__init__()

        self.layers = layers
        self.splits = splits
        self.survival = nn.Sequential(*create_nn(inputdim, layers + [len(splits) + 1])[:-1])

    def forward(self, x, labels = None):
        return {'logits': self.survival(x)}