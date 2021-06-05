import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)

# This class is build for loading different datasets in super resolution tasks
class Sr_datasets(Dataset):
    def __init__(self, data_root, data_dict, transform, load_all=True, to_gray=False, s_factor=1, repeat_crop=1):
        self.data_root = data_root
        self.transform = transform
        self.load_all = load_all
        self.s_factor = s_factor
        self.repeat_crop = repeat_crop
        if self.load_all is False:
            self.data_dict = data_dict
        else:
            self.data_dict = []
            for sample_info in data_dict:
                hr_img = Image.open('/'.join((self.data_root, sample_info['gt_path']))).copy()
                lw_img = Image.open('/'.join((self.data_root, sample_info['x{}_path'.format(self.s_factor)]))).copy()
                if hr_img.mode != 'RGB':
                    hr_img, lw_img = hr_img.convert('RGB'), lw_img.convert('RGB')
                [width, height] = sample_info['x{}_size'.format(self.s_factor)]
                sample = {
                    'hr_img': hr_img,
                    'lw_img': lw_img,
                    'width': width,
                    'height': height
                }
                self.data_dict.append(sample)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        sample_info = self.data_dict[idx]
        if self.load_all is False:
            image = Image.open('/'.join((self.data_root, sample_info['x{}_path'.format(self.s_factor)])))
            target = Image.open('/'.join((self.data_root, sample_info['gt_path'])))
            if image.mode != 'RGB':
                image, target = image.convert('RGB'), target.convert('RGB')
        else:
            image = sample_info['lw_img']
            target = sample_info['hr_img']

        sample = {'image': image, 'target': target}

        if self.repeat_crop != 1:
            image_stacks = []
            target_stacks = []

            for i in range(self.repeat_crop):
                sample_patch = self.transform(sample)
                image_stacks.append(sample_patch['image'])
                target_stacks.append(sample_patch['target'])
            return torch.stack(image_stacks), torch.stack(target_stacks)

        else:
            sample_patch = self.transform(sample)
            return sample_patch['image'], sample_patch['target']
