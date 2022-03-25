import os, sys
import time
import torch

import numpy as np
import matplotlib.pyplot as plt

from net_module import loss_functions as loss_func
from network_manager import NetworkManager
from data_handle import data_handler as dh

from util import utils_yaml

import pickle
from datetime import datetime

def check_device():
    if torch.cuda.is_available():
        print('GPU count:', torch.cuda.device_count(),
              'Current 1:', torch.cuda.current_device(), torch.cuda.get_device_name(0))
    else:
        print(f'CUDA not working! Pytorch: {torch.__version__}.')
        sys.exit(0)
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)

def load_param(root_dir, config_file, param_in_list=True):
    if param_in_list:
        param_list = utils_yaml.from_yaml_all(os.path.join(root_dir, 'Config/', config_file))
        return {**param_list[0], **param_list[1], **param_list[2], **param_list[3]}
    else:
        return utils_yaml.from_yaml(os.path.join(root_dir, 'Config/', config_file))

def load_path(param, root_dir, zip=True):
    if zip:
        save_path = os.path.join(root_dir, param['model_path'])
        zip_path  = os.path.join(root_dir, param['zip_path'])
        csv_path  = os.path.join(param['data_name'], param['label_csv'])
        data_dir  = param['data_name']
        return save_path, csv_path, data_dir, zip_path
    else:
        save_path = os.path.join(root_dir, param['model_path'])
        csv_path  = os.path.join(root_dir, param['label_path'])
        data_dir  = os.path.join(root_dir, param['data_path'])
        return save_path, csv_path, data_dir

def load_data(param, paths, Dataset, transform, validation_cache=10, zip=True, load_for_test=False):
    if zip:
        myDS = Dataset(zip_path=paths[3], csv_path=paths[1], root_dir=paths[2], channel_per_image=param['cpi'], transform=transform)
    else:
        myDS = Dataset(csv_path=paths[1], root_dir=paths[2], channel_per_image=param['cpi'], transform=transform)
    myDH = dh.DataHandler(myDS, batch_size=param['batch_size'], validation_prop=param['validation_prop'], validation_cache=validation_cache)
    if not load_for_test:
        print("Data prepared. #Samples(training, val):{}, #Batches:{}".format(myDH.return_length_ds(), myDH.return_length_dl()))
    print('Sample: {\'image\':',myDS[0]['image'].shape,'\'label\':',myDS[0]['label'],'}')
    return myDS, myDH

def load_manager(param, Net, loss):
    net = Net(param['input_channel'])
    myNet = NetworkManager(net, loss, early_stopping=param['early_stopping'], device=param['device'])
    myNet.build_Network()

    ### Label process
    x_grid = np.arange(0, param['x_max_px']+1, param['cell_width'])
    y_grid = np.arange(0, param['y_max_px']+1, param['cell_width'])
    myNet.label_process(x_range=param['x_range'], y_range=param['y_range'], 
                        x_max_px=param['x_max_px'], y_max_px=param['y_max_px'], 
                        x_grid=x_grid, y_grid=y_grid)
    return myNet

def save_profile(manager, save_path='./'):
    now = datetime.now()
    dt = now.strftime("%d_%m_%Y__%H_%M_%S")

    manager.plot_history_loss()
    plt.savefig(os.path.join(save_path, dt+'.png'), bbox_inches='tight')
    plt.close()

    loss_dict = {'loss':manager.Loss, 'val_loss':manager.Val_loss}
    with open(os.path.join(save_path, dt+'.pickle'), 'wb') as pf:
        pickle.dump(loss_dict, pf)

def main_train(root_dir, config_file, Dataset, zip:bool, transform, Net, loss):
    ### Check and load
    check_device()
    param = load_param(root_dir, config_file)
    paths = load_path(param, root_dir, zip)
    _, myDH = load_data(param, paths, Dataset, transform, zip=zip)
    myNet = load_manager(param, Net, loss)
    myNet.build_Network()

    ### Training
    start_time = time.time()
    myNet.train(myDH, param['batch_size'], param['epoch'], val_after_batch=param['batch_size'])
    total_time = round((time.time()-start_time)/3600, 4)
    if (paths[0] is not None) & myNet.complete:
        torch.save(myNet.model.state_dict(), paths[0])
    nparams = sum(p.numel() for p in myNet.model.parameters() if p.requires_grad)
    print("\nTraining done: {} parameters. Cost time: {}h.".format(nparams, total_time))

    save_profile(myNet)

def main_test_pre(root_dir, config_file, Dataset, zip:bool, transform, Net):
    ### Check and load
    param = load_param(root_dir, config_file)
    paths = load_path(param, root_dir, zip)
    myDS, _ = load_data(param, paths, Dataset, transform, zip=zip, load_for_test=True)
    myNet = load_manager(param, Net, {})
    myNet.build_Network()
    myNet.model.load_state_dict(torch.load(paths[0]))
    myNet.model.eval() # with BN layer, must run eval first
    return myDS, myNet

def main_test(dataset, net, idx, y_flip=False):
    img   = dataset[idx]['image']
    label = dataset[idx]['label']
    traj  = dataset[idx]['traj']
    index = dataset[idx]['index']
    pred = net.inference(img)

    traj = torch.tensor(traj)
    traj = loss_func.convert_coords2px(traj, net.x_range, net.y_range, net.x_max_px, net.y_max_px, y_flip=y_flip)
    px_idx   = loss_func.convert_coords2px(label.unsqueeze(0), net.x_range, net.y_range, net.x_max_px, net.y_max_px, y_flip=y_flip)
    cell_idx = loss_func.convert_px2cell(px_idx, net.x_grid, net.y_grid, device='cpu') # (xmin ymin xmax ymax)
    return img, px_idx, cell_idx, traj, index, pred