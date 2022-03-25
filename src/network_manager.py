import os, sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from net_module import loss_functions as loss_func

from timeit import default_timer as timer
from datetime import timedelta

class NetworkManager():
    """ 
    
    """
    def __init__(self, net, loss_dict, early_stopping=0, device='cuda', checkpoint_dir=None, verbose=True):
        self.vb = verbose
        
        self.lr = 1e-4      # learning rate
        self.w_decay = 1e-5 # L2 regularization

        self.Loss = []      # track the loss
        self.Val_loss= []   # track the validation loss
        self.Val_metric = [] # track the closest component's loss
        self.es = early_stopping

        self.net = net
        self.loss_function = loss_dict['loss']
        self.metric = loss_dict['metric']
        self.device = device
        self.save_dir = checkpoint_dir

        self.complete = False

        self.label_process_done = False
        self.training_time(None, None, None, init=True)

    def training_time(self, remaining_epoch, remaining_batch, batch_per_epoch, init=False):
        if init:
            self.batch_time = []
            self.epoch_time = []
        else:
            batch_time_average = sum(self.batch_time)/max(len(self.batch_time),1)
            if len(self.epoch_time) == 0:
                epoch_time_average = batch_time_average * batch_per_epoch
            else:
                epoch_time_average = sum(self.epoch_time)/max(len(self.epoch_time),1)
            eta = round(epoch_time_average * remaining_epoch + batch_time_average * remaining_batch, 1)
            return timedelta(seconds=batch_time_average), timedelta(seconds=epoch_time_average), timedelta(seconds=eta)

    def label_process(self, x_range, y_range, x_max_px, y_max_px, x_grid, y_grid, y_flip=False):
        self.x_range = x_range
        self.y_range = y_range
        self.x_max_px = x_max_px
        self.y_max_px = y_max_px
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.y_flip = y_flip
        self.label_process_done = True

    def build_Network(self):
        self.gen_Model()
        self.gen_Optimizer(self.model.parameters())
        if self.vb:
            print(self.model)

    def gen_Model(self):
        self.model = nn.Sequential()
        self.model.add_module('Net', self.net)
        if self.device == 'multi':
            self.model = nn.DataParallel(self.model.to(torch.device("cuda:0")))
        elif self.device == 'cuda':
            self.model = self.model.to(torch.device("cuda:0"))
        elif self.device == 'cpu': 
            pass
        else:
            raise ModuleNotFoundError(f'No such device as {self.device} (should be "multi", "cuda", or "cpu").')
        return self.model

    def gen_Optimizer(self, parameters):
        self.optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.w_decay, betas=(0.99, 0.999))
        # self.optimizer = optim.SGD(parameters, lr=1e-3, momentum=0.9)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        return self.optimizer

    def inference(self, data):
        if self.device in ['multi', 'cuda']:
            device = torch.device("cuda:0")
        else:
            device = 'cpu'
        with torch.no_grad():
            e_grid = self.model(data.unsqueeze(0).float().to(device))
            e_grid = e_grid[0].cpu().detach().numpy()
        return e_grid

    def validate(self, data, labels, device='cuda', loss_function=None):
        if loss_function is None:
            loss_function = self.loss_function
        outputs = self.model(data)
        loss = loss_function(outputs, labels, device)
        return loss

    def train_batch(self, batch, label, device='cuda', loss_function=None):
        self.model.zero_grad()
        loss = self.validate(batch, label, device, loss_function)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, data_handler, batch_size, epoch, val_after_batch=1):
        print('\nTraining...')
        if self.device in ['multi', 'cuda']:
            device = torch.device("cuda:0")
        else:
            device = 'cpu'

        assert(self.label_process_done),('Process the label first.')
        data_val = data_handler.dataset_val
        max_cnt_per_epoch = data_handler.return_length_dl()
        min_val_loss = np.Inf
        min_val_loss_epoch = np.Inf
        epochs_no_improve = 0
        cnt = 0 # counter for batches over all epochs
        for ep in range(epoch):
            epoch_time_start = timer() ### TIMER

            cnt_per_epoch = 0 # counter for batches within the epoch

            loss_epoch = self.loss_function

            while (cnt_per_epoch<max_cnt_per_epoch):
                cnt += 1
                cnt_per_epoch += 1

                batch_time_start = timer() ### TIMER

                batch, label = data_handler.return_batch()
                batch, label = batch.float().to(device), label.float().to(device)

                if (self.x_range!=0) & (self.y_range!=0):
                    label = loss_func.convert_coords2px(label, self.x_range, self.y_range, self.x_max_px, self.y_max_px, y_flip=self.y_flip)

                # print(batch.shape)
                # ### XXX
                # plt.figure()
                # plt.imshow(batch[0,0,:,:].cpu())
                # plt.plot(label[0,0].cpu(), label[0,1].cpu(), 'rx')
                # plt.show()

                # _, [ax1, ax2] = plt.subplots(1,2)
                # ax1.imshow(batch[0,0,:,:].cpu())
                # ax1.set_xticks(self.x_grid)
                # ax1.set_yticks(self.y_grid)
                # ax1.grid(linestyle=':')
                # ax1.plot(label.cpu()[0,0], label.cpu()[0,1], 'rx')
                # ax2.imshow(batch[0,1,:,:].cpu(), cmap='gray')
                # ax2.set_xticks(self.x_grid)
                # ax2.set_yticks(self.y_grid)
                # ax2.grid(linestyle=':')
                # ax2.plot(label.cpu()[0,0], label.cpu()[0,1], 'rx')
                # plt.show()
                # sys.exit(0)
                # ###

                label = loss_func.convert_px2cell(label, self.x_grid, self.y_grid, device=device)

                loss = self.train_batch(batch, label, device=device, loss_function=loss_epoch) # train here
                self.Loss.append(loss.item())

                if len(data_val)>0 & (cnt_per_epoch%val_after_batch==0):
                    del batch
                    del label
                    val_data, val_label = data_handler.return_val()
                    val_data, val_label = val_data.float().to(device), val_label.float().to(device)

                    val_label = loss_func.convert_coords2px(val_label, self.x_range, self.y_range, self.x_max_px, self.y_max_px)
                    val_label = loss_func.convert_px2cell(val_label, self.x_grid, self.y_grid)

                    val_loss = self.validate(val_data, val_label)
                    self.Val_loss.append((cnt, val_loss.item()))
                    del val_data
                    del val_label
                    if val_loss < min_val_loss_epoch:
                        min_val_loss_epoch = val_loss

                self.batch_time.append(timer()-batch_time_start)  ### TIMER

                if np.isnan(loss.item()): # assert(~np.isnan(loss.item())),("Loss goes to NaN!")
                    print(f"Loss goes to NaN! Fail after {cnt} batches.")
                    self.complete = False
                    return

                if (cnt_per_epoch%20==0 or cnt_per_epoch==max_cnt_per_epoch) & (self.vb):
                    _, _, eta = self.training_time(epoch-ep-1, max_cnt_per_epoch-cnt_per_epoch, max_cnt_per_epoch) # TIMER
                    if len(data_val)>0:
                        prt_loss = f'Loss/Val_loss: {round(loss.item(),4)}/{round(val_loss.item(),4)}'
                    else:
                        prt_loss = f'Training loss: {round(loss.item(),4)}'
                    prt_num_samples = f'{cnt_per_epoch*batch_size/1000}k/{max_cnt_per_epoch*batch_size/1000}k'
                    prt_num_epoch = f'Epoch {ep+1}/{epoch}'
                    prt_eta = f'ETA {eta}'
                    print('\r'+prt_loss+', '+prt_num_samples+', '+prt_num_epoch+', '+prt_eta, end='    ')

            if min_val_loss_epoch < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = min_val_loss_epoch
            else:
                epochs_no_improve += 1
            if (self.es > 0) & (epochs_no_improve >= self.es):
                print(f'\nEarly stopping after {self.es} epochs with no improvement.')
                break

            self.epoch_time.append(timer()-epoch_time_start)  ### TIMER
            self.lr_scheduler.step()
            print() # end while
        self.complete = True
        # plt.show()
        print('\nTraining Complete!')


    def plot_history_loss(self):
        plt.figure()
        if len(self.Val_loss):
            plt.plot(np.array(self.Val_loss)[:,0], np.array(self.Val_loss)[:,1], '.', label='val_loss')
        plt.plot(np.linspace(1,len(self.Loss),len(self.Loss)), self.Loss, '.', label='loss')
        plt.xlabel('#batch')
        plt.ylabel('Loss')
        plt.legend()
