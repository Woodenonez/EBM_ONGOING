import os, sys
from pathlib import Path

import torch
import torchvision

from net_module import loss_functions as loss_func
from net_module.net import GridCodec
from data_handle import data_handler as dh
from data_handle import dataset as ds

import pre_load

print("Program: training\n")

### Config
root_dir = Path(__file__).parents[1]

config_file = 'ebm_sdd1v_train.yml'
loss = {'loss': loss_func.loss_nll, 'metric': loss_func.loss_mae}
param = pre_load.load_param(root_dir, config_file)

data_from_zip = True
composed = torchvision.transforms.Compose([dh.Rescale((param['y_max_px'], param['x_max_px'])), 
                                           dh.ToTensor()])
Dataset = ds.ImageStackDatasetSDD_ZIP
Net = GridCodec

### Training
pre_load.main_train(root_dir, config_file, Dataset=Dataset, Net=Net, 
                    zip=data_from_zip, transform=composed, loss=loss)