import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from net_module.net import GridCodec
from data_handle import data_handler as dh
from data_handle import dataset as ds
from data_handle import mid_object
from net_module import loss_functions as loss_func

import pre_load
from util import utils_test

print("Program: training\n")
if torch.cuda.is_available():
    print(torch.cuda.current_device(),torch.cuda.get_device_name(0))
else:
    print(f'CUDA not working! Pytorch: {torch.__version__}.')
    sys.exit(0)
torch.cuda.empty_cache()

### Config
root_dir = Path(__file__).parents[1]
config_file = 'ebm_sdd3v_test.yml'
param = pre_load.load_param(root_dir, config_file)

data_from_zip = True
composed = torchvision.transforms.Compose([dh.Rescale((param['y_max_px'], param['x_max_px'])), 
                                           dh.ToTensor()])
Dataset = ds.ImageStackDatasetSDD_ZIP
Net = GridCodec

### Prepare
dataset, net = pre_load.main_test_pre(root_dir, config_file, Dataset, data_from_zip, composed, Net)

### Visualization option
idx_start = 0
idx_end = len(dataset)
pause_time = 0.1

### Visualize
fig, axes = plt.subplots(1,3)
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
for idx in idc:

    print(idx)

    [ax.cla() for ax in axes]
    
    img, px_idx, cell_idx, traj, index, e_grid = pre_load.main_test(dataset, net, idx=idx)
    e_grid = e_grid[0,:,:] # if e_grid has only 1 channel
    # e_min  = np.min(e_grid)
    # e_max  = np.max(e_grid)
    # e_grid = (e_grid-e_min)/(e_max-e_min)
    prob_map = loss_func.convert_grid2prob(torch.tensor(e_grid), threshold=0.3, temperature=0.5)

    utils_test.plot_on_sdd(axes, img[-2,:,:], px_idx[0,:], cell_idx, traj, e_grid, prob_map)
    # boundary_coords, obstacle_list, _ = return_Map(index)
    # graph = Graph(boundary_coords, obstacle_list)
    # graph.plot_map(ax1, clean=1)

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    
    if pause_time==0:
        plt.pause(0.1)
        input()
    else:
        plt.pause(pause_time)

plt.show()
