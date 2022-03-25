import os, sys

import numpy as np

import torch
from torch import nn
from torch import tensor as ts
from torch.nn.init import kaiming_normal_, constant_

def compact_linear_layer(with_batch_norm, dim_input, dim_output):
    if with_batch_norm:
        layer = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.BatchNorm1d(dim_output),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    else:
        layer = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    return layer

def compact_conv_layer(with_batch_norm, input_channel, output_channel, kernel_size=3, stride=1, padding=0, activate=True):
    if with_batch_norm & activate:
        layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    elif ~with_batch_norm & activate:
        layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    elif with_batch_norm & ~activate:
        layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channel)
        )
    else:
        raise(Exception('No need to use compact layers.'))
    return layer

