from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np 
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ignore_label = 255
ID_TO_TRAINID = {153: 0, 204: 1, 0: 2, 102: 3, 51: 4}
#ID_TO_TRAINID = {0: 0, 153: 1, 204: 2, 0: 3, 102: 4, 51: 5}

class UAVDataset(BaseDataSet):
    def __init__(self, mode='fine', **kwarg):
        self.num_classes = 5
        self.palette = palette.UAV_palette
        self.id_to_trainid = ID_TO_TRAINID
        super(UAVDataset, self).__init__(**kwarg)

    def _set_files(self):
        SUFIX_IMG = '_img.png'
        SUFIX_LABEL = '_lab.png'
        img_dir_name = 'uav-images'
        img_path = os.path.join(self.root, img_dir_name, self.split, 'org_img')
        label_path = os.path.join(self.root, img_dir_name, self.split, 'label_img')
        #assert os.listdir(img_path) == os.listdir(label_path)

        img_paths, label_paths = [], []
        img_paths.extend(sorted(glob(os.path.join(img_path, f'*{SUFIX_IMG}'))))
        label_paths.extend(sorted(glob(os.path.join(label_path, f'*{SUFIX_LABEL}'))))
        # @sh: add
        print(len(img_paths))
        print(len(label_paths))
        self.files = list(zip(img_paths,  label_paths))

    def _load_data(self, index):
        img_path, label_path = self.files[index]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.float32)

        for k, v in self.id_to_trainid.items():
            label[ label == k] = v

        return img, label, img_id


class UAV(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, mode='fine', val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.28689529, 0.32513294, 0.28389176] # how to calculate them for uavdataset??
        self.STD = [0.17613647, 0.18099176, 0.17772235] # how to calculate them for uavdataset??

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = UAVDataset(mode=mode, **kwargs)
        super(UAV, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

