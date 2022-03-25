import os
import glob

import numpy as np
import pandas as pd
from skimage import io

import torch
from torch.utils.data import Dataset

import zipfile
from util import utils_np

'''
'''

class ImageStackDatasetSim(Dataset):
    def __init__(self, csv_path, root_dir, channel_per_image, transform=None, T_channel=False):
        '''
        Args:
            csv_path: Path to the CSV file with dataset info.
            root_dir: Directory with all image folders.
                      root_dir - obj_folder - obj & env
        '''
        super().__init__()
        self.info_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.tr = transform
        self.with_T = T_channel
        self.cpi = channel_per_image

        self.nc = len(list(self.info_frame))-4 # number of image channels in total
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        self.T = info['T']
        index = info['index']
        traj = []
        for i in range(self.nc):
            img_name = info['f{}'.format(i)]
            obj_id = img_name.split('_')[0]

            if self.cpi == 1:
                img_path = os.path.join(self.root_dir, obj_id, img_name)
                this_x = float(img_name[:-4].split('_')[2])
                this_y = float(img_name[:-4].split('_')[3])
                traj.append([this_x,this_y])
            elif self.cpi == 2:
                if len(img_name.split('_'))==5:
                    img_path = os.path.join(self.root_dir, obj_id, 'obj', img_name)
                    time_step = int(img_name[:-4].split('_')[1])
                    this_x = float(img_name[:-4].split('_')[2])
                    this_y = float(img_name[:-4].split('_')[3])
                    traj.append([this_x,this_y])
                else:
                    img_path = os.path.join(self.root_dir, obj_id, 'env', img_name)

            image = self.togray(io.imread(img_path))
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)

        if self.with_T:
            T_channel = np.ones(shape=[self.img_shape[0],self.img_shape[1],1])*self.T # T_channel
            input_img = np.concatenate((input_img, T_channel), axis=2)                # T_channel

        label = {'x':info['x'], 'y':info['y']}
        sample = {'image':input_img, 'label':label}

        if self.tr:
            sample = self.tr(sample)

        sample['index'] = index
        sample['traj'] = traj
        if self.cpi == 2:
            sample['time'] = time_step

        return sample

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        info = self.info_frame.iloc[0]
        img_name = info['f0']
        obj_id = img_name.split('_')[0]

        if self.cpi == 1:
            img_path = os.path.join(self.root_dir, obj_id, img_name)
        elif self.cpi == 2:
            img_path = os.path.join(self.root_dir, obj_id, 'obj', img_name)

        image = self.togray(io.imread(img_path))
        return image.shape

class ImageStackDatasetSim_ZIP(Dataset):
    def __init__(self, zip_path, csv_path, root_dir, channel_per_image, transform=None, T_channel=False):
        '''
        Args:
            zip_path: Path to the ZIP file with everything
            csv_path: Path to the CSV file with dataset info.
            root_dir: Directory with all image folders.
                      root_dir - obj_folder - obj & env
        '''
        super().__init__()
        self.archive = zipfile.ZipFile(zip_path, 'r')

        self.info_frame = pd.read_csv(self.archive.open(csv_path))
        self.root_dir = root_dir
        self.tr = transform
        self.with_T = T_channel
        self.cpi = channel_per_image

        self.nc = len(list(self.info_frame))-4 # number of image channels in total
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        self.T = info['T']
        index = info['index']
        traj = []
        for i in range(self.nc):
            img_name = info['f{}'.format(i)]
            obj_id = img_name.split('_')[0]

            if self.cpi == 1:
                img_path = os.path.join(self.root_dir, obj_id, img_name)
                this_x = float(img_name[:-4].split('_')[2])
                this_y = float(img_name[:-4].split('_')[3])
                traj.append([this_x,this_y])
            elif self.cpi == 2:
                if len(img_name.split('_'))==5:
                    img_path = os.path.join(self.root_dir, obj_id, 'obj', img_name)
                    time_step = int(img_name[:-4].split('_')[1])
                    this_x = float(img_name[:-4].split('_')[2])
                    this_y = float(img_name[:-4].split('_')[3])
                    traj.append([this_x,this_y])
                else:
                    img_path = os.path.join(self.root_dir, obj_id, 'env', img_name)

            image = self.togray(io.imread(self.archive.open(img_path)))
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)

        if self.with_T:
            T_channel = np.ones(shape=[self.img_shape[0],self.img_shape[1],1])*self.T # T_channel
            input_img = np.concatenate((input_img, T_channel), axis=2)                # T_channel

        label = {'x':info['x'], 'y':info['y']}
        sample = {'image':input_img, 'label':label}

        if self.tr:
            sample = self.tr(sample)

        sample['index'] = index
        sample['traj'] = traj
        if self.cpi == 2:
            sample['time'] = time_step

        return sample

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        info = self.info_frame.iloc[0]
        img_name = info['f0']
        obj_id = img_name.split('_')[0]

        if self.cpi == 1:
            img_path = os.path.join(self.root_dir, obj_id, img_name)
        elif self.cpi == 2:
            img_path = os.path.join(self.root_dir, obj_id, 'obj', img_name)

        image = self.togray(io.imread(self.archive.open(img_path)))
        return image.shape


class ImageStackDatasetSDD(Dataset):
    def __init__(self, csv_path, root_dir, ext='.jpg', channel_per_image=None, transform=None, T_channel=False):
        '''
        Args:
            csv_path: Path to the CSV file with dataset info.
            root_dir: Directory with all image folders.
                      root_dir - video_folder - imgs
        '''
        super().__init__()
        self.info_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.tr = transform
        self.with_T = T_channel
        self.ext = ext

        self.nc = len(list(self.info_frame))-5 # number of image channels in half
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        self.T = info['T']
        index = info['index']
        traj = []
        for i in range(self.nc):
            img_name = info[f't{i}'].split('_')[0] + self.ext
            video_idx = info['index']

            img_path = os.path.join(self.root_dir, video_idx, img_name)

            csv_name = glob.glob(os.path.join(self.root_dir, video_idx, '*.csv'))
            original_scale = os.path.basename(csv_name[0]).split('.')[0]
            original_scale = (int(original_scale.split('_')[0]), int(original_scale.split('_')[1])) # HxW

            time_step = int(info[f't{i}'].split('_')[0])
            this_x = float(info[f't{i}'].split('_')[1])
            this_y = float(info[f't{i}'].split('_')[2])
            traj.append([this_x,this_y])

            image = self.togray(io.imread(img_path))
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)

            white_canvas = np.zeros_like(image)
            # obj_coords = self.rescale_label((this_x, this_y), original_scale)
            obj_coords = (this_x, this_y)
            obj_map = utils_np.np_gaudist_map(obj_coords, white_canvas, sigmas=[20,20])
            input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)

        if self.with_T:
            T_channel = np.ones(shape=[self.img_shape[0],self.img_shape[1],1])*self.T # T_channel
            input_img = np.concatenate((input_img, T_channel), axis=2)                # T_channel

        label = {'x':info['x'], 'y':info['y']}
        sample = {'image':input_img, 'label':label}

        if self.tr:
            sample = self.tr(sample)

        sample['index'] = index
        sample['traj'] = traj
        sample['time'] = time_step

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        info = self.info_frame.iloc[0]
        img_name = info['t0'].split('_')[0] + self.ext
        video_folder = info['index']
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(img_path))
        return image.shape

class ImageStackDatasetSDD_ZIP(Dataset):
    def __init__(self, zip_path, csv_path, root_dir, ext='.jpg', channel_per_image=None, transform=None, T_channel=False):
        '''
        Args:
            zip_path: Path (absolute) to the ZIP file with everything
            csv_path: Path (relative) to the CSV file with dataset info.
            root_dir: Directory (relative) with all image folders.
                      root_dir - obj_folder - obj & other
        '''
        super().__init__()
        self.archive = zipfile.ZipFile(zip_path, 'r')

        self.info_frame = pd.read_csv(self.archive.open(csv_path))
        self.root_dir = root_dir
        self.tr = transform
        self.with_T = T_channel
        self.cpi = channel_per_image
        self.ext = ext

        self.nc = len(list(self.info_frame))-5 # number of image channels in half
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        self.T = info['T']
        index = info['index']
        traj = []
        for i in range(self.nc):
            img_name = info[f't{i}'].split('_')[0] + self.ext
            video_idx = info['index']

            img_path = os.path.join(self.root_dir, video_idx, img_name)

            csv_name = [x for x in self.archive.namelist() if ((video_idx in x)&('csv' in x))]
            original_scale = os.path.basename(csv_name[0]).split('.')[0]
            original_scale = (int(original_scale.split('_')[0]), int(original_scale.split('_')[1])) # HxW

            time_step = int(info[f't{i}'].split('_')[0])
            this_x = float(info[f't{i}'].split('_')[1])
            this_y = float(info[f't{i}'].split('_')[2])
            traj.append([this_x,this_y])

            image = self.togray(io.imread(self.archive.open(img_path)))
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)

            white_canvas = np.zeros_like(image)
            # obj_coords = self.rescale_label((this_x, this_y), original_scale)
            obj_coords = (this_x, this_y)
            obj_map = utils_np.np_gaudist_map(obj_coords, white_canvas, sigmas=[20,20])
            input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)

        if self.with_T:
            T_channel = np.ones(shape=[self.img_shape[0],self.img_shape[1],1])*self.T # T_channel
            input_img = np.concatenate((input_img, T_channel), axis=2)                # T_channel

        label = {'x':info['x'], 'y':info['y']}
        sample = {'image':input_img, 'label':label}

        if self.tr:
            sample = self.tr(sample)

        sample['index'] = index
        sample['traj'] = traj
        sample['time'] = time_step

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        info = self.info_frame.iloc[0]
        img_name = info['t0'].split('_')[0] + self.ext
        video_folder = info['index']
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(self.archive.open(img_path)))
        return image.shape


class ImageStackDatasetSDDtr(Dataset): # for trajectory
    def __init__(self, csv_path, root_dir, ext='.jpg', transform=None, T_channel=None):
        '''
        Args:
            csv_path: Path to the CSV file with dataset info.
            root_dir: Directory with all image folders.
                      root_dir - video_folder - imgs
        '''
        super().__init__()
        self.info_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.tr = transform
        self.ext = ext

        self.nc = len([x for x in list(self.info_frame) if 't' in x]) # number of image channels in half
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        index = info['index']
        traj = []
        for i in range(self.nc):
            img_name = info[f't{i}'].split('_')[0] + self.ext
            video_idx = info['index']

            img_path = os.path.join(self.root_dir, video_idx, img_name)

            csv_name = glob.glob(os.path.join(self.root_dir, video_idx, '*.csv'))
            original_scale = os.path.basename(csv_name[0]).split('.')[0]
            original_scale = (int(original_scale.split('_')[0]), int(original_scale.split('_')[1])) # HxW

            time_step = int(info[f't{i}'].split('_')[0])
            this_x = float(info[f't{i}'].split('_')[1])
            this_y = float(info[f't{i}'].split('_')[2])
            traj.append([this_x,this_y])

            image = self.togray(io.imread(img_path))
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)

            white_canvas = np.zeros_like(image)
            # obj_coords = self.rescale_label((this_x, this_y), original_scale)
            obj_coords = (this_x, this_y)
            obj_map = utils_np.np_gaudist_map(obj_coords, white_canvas, sigmas=[20,20])
            input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)

        label_name_list = [x for x in list(self.info_frame) if 'T' in x]
        label_list = list(info[label_name_list].values)
        label_list = [(float(x.split('_')[0]), float(x.split('_')[1])) for x in label_list]
        label = dict(zip(label_name_list, label_list))
        sample = {'image':input_img, 'label':label}

        if self.tr:
            sample = self.tr(sample)

        sample['index'] = index
        sample['traj'] = traj
        sample['time'] = time_step

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        info = self.info_frame.iloc[0]
        img_name = info['t0'].split('_')[0] + self.ext
        video_folder = info['index']
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(img_path))
        return image.shape

class ImageStackDatasetSDDtr_ZIP(Dataset): # for trajectory
    def __init__(self, zip_path, csv_path, root_dir, ext='.jpg', transform=None, T_channel=None):
        '''
        Args:
            zip_path: Path (absolute) to the ZIP file with everything
            csv_path: Path (relative) to the CSV file with dataset info.
            root_dir: Directory (relative) with all image folders.
                      root_dir - obj_folder - obj & other
        '''
        super().__init__()
        self.archive = zipfile.ZipFile(zip_path, 'r')

        self.info_frame = pd.read_csv(self.archive.open(csv_path))
        self.root_dir = root_dir
        self.tr = transform
        self.ext = ext

        self.nc = len([x for x in list(self.info_frame) if 't' in x]) # number of image channels in half
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        index = info['index']
        traj = []
        for i in range(self.nc):
            img_name = info[f't{i}'].split('_')[0] + self.ext
            video_idx = info['index']

            img_path = os.path.join(self.root_dir, video_idx, img_name)

            csv_name = [x for x in self.archive.namelist() if ((video_idx in x)&('csv' in x))]
            original_scale = os.path.basename(csv_name[0]).split('.')[0]
            original_scale = (int(original_scale.split('_')[0]), int(original_scale.split('_')[1])) # HxW

            time_step = int(info[f't{i}'].split('_')[0])
            this_x = float(info[f't{i}'].split('_')[1])
            this_y = float(info[f't{i}'].split('_')[2])
            traj.append([this_x,this_y])

            image = self.togray(io.imread(self.archive.open(img_path)))
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)

            white_canvas = np.zeros_like(image)
            # obj_coords = self.rescale_label((this_x, this_y), original_scale)
            obj_coords = (this_x, this_y)
            obj_map = utils_np.np_gaudist_map(obj_coords, white_canvas, sigmas=[20,20])
            input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)

        label_name_list = [x for x in list(self.info_frame) if 'T' in x]
        label_list = list(info[label_name_list].values)
        label_list = [(float(x.split('_')[0]), float(x.split('_')[1])) for x in label_list]
        label = dict(zip(label_name_list, label_list))
        sample = {'image':input_img, 'label':label}

        if self.tr:
            sample = self.tr(sample)

        sample['index'] = index
        sample['traj'] = traj
        sample['time'] = time_step

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        info = self.info_frame.iloc[0]
        img_name = info['t0'].split('_')[0] + self.ext
        video_folder = info['index']
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(self.archive.open(img_path)))
        return image.shape


class MaskStackWithSegSDD(Dataset):
    def __init__(self, csv_path, seg_path, channel_per_image=None, transform=None, T_channel=False):
        '''
        Args:
            csv_path: Path to the CSV file with dataset info.
            seg_path: Path to the segmentation image.
        '''
        super().__init__()
        self.info_frame = pd.read_csv(csv_path)
        self.seg_path = seg_path
        self.tr = transform
        self.with_T = T_channel

        self.nc = len(list(self.info_frame))-5 # number of image channels in half
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        self.T = info['T']
        index = info['index']
        traj = []

        img_path = self.seg_path
        image = self.togray(io.imread(img_path))
        input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)
        for i in range(self.nc):
            # img_name = info[f't{i}'].split('_')[0] + self.ext
            # video_idx = info['index']

            time_step = int(info[f't{i}'].split('_')[0])
            this_x = float(info[f't{i}'].split('_')[1])
            this_y = float(info[f't{i}'].split('_')[2])
            traj.append([this_x,this_y])

            white_canvas = np.zeros_like(image)
            obj_coords = (this_x, this_y)
            obj_map = utils_np.np_gaudist_map(obj_coords, white_canvas, sigmas=[20,20])
            input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)
        

        if self.with_T:
            T_channel = np.ones(shape=[self.img_shape[0],self.img_shape[1],1])*self.T # T_channel
            input_img = np.concatenate((input_img, T_channel), axis=2)           # T_channel

        label = {'x':info['x'], 'y':info['y']}
        sample = {'image':input_img, 'label':label}

        if self.tr:
            sample = self.tr(sample)

        sample['index'] = index
        sample['traj'] = traj
        sample['time'] = time_step

        return sample

    def togray(self, image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self):
        image = self.togray(io.imread(self.seg_path))
        return image.shape

