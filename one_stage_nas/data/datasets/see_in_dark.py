import os
import json
import torch
import rawpy
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


# the dataloader of seeing in the dark on the Sony dataset
class Sid_dataset(Dataset):
    def __init__(self, data_root, data_dict, transform, load_all=False, to_gray=False, s_factor=1, repeat_crop=1):
        self.data_root = data_root
        self.transform = transform
        self.load_all = load_all
        self.to_gray = to_gray
        self.repeat_crop = repeat_crop
        self.load_all = load_all

        if not self.load_all:
            self.data_dict = data_dict
        else:
            self.data_dict = []
            for sample_info in data_dict:
                raw_data = []
                for raw_path in sample_info['raw_path']:
                    raw_data.append(rawpy.imread('/'.join((self.data_root, raw_path))))
                gt_data =rawpy.imread('/'.join((self.data_root, sample_info['gt_path'])))

                sample = {
                    'sample_id': sample_info['sample_id'],
                    'raw_data': raw_data,
                    'gt_data': gt_data,
                    'raw_exposure':sample_info['raw_exposure'],
                    'gt_exposure': sample_info['gt_exposure'],
                }

                self.data_dict.append(sample)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        sample_info = self.data_dict[idx]
        raw_exposure = sample_info['raw_exposure']
        gt_exposure = sample_info['gt_exposure']

        raw_index = random.randint(0, len(raw_exposure)-1)
        raw_exposure_cur = raw_exposure[raw_index]

        if not self.load_all:
            raw_path = sample_info['raw_path'][raw_index]
            gt_path = sample_info['gt_path']
            raw_input = rawpy.imread('/'.join((self.data_root, raw_path)))
            gt_input = rawpy.imread('/'.join((self.data_root, gt_path)))
        else:
            raw_input = sample_info['raw_data'][raw_index]
            gt_input = sample_info['gt_data']

        arw_input, width, height = self.pack_raw(raw_input)
        rgb_input = raw_input.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        rgb_input = (rgb_input / 65535.0).astype(np.float32)

        gt_input = gt_input.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_input = (gt_input / 65535.0).astype(np.float32)

        ratio = min(gt_exposure/raw_exposure_cur, 300)
        arw_input = np.minimum(np.maximum(arw_input * ratio, 0), 1)
        rgb_input = np.minimum(np.maximum(rgb_input * ratio, 0), 1)
        gt_input = np.minimum(np.maximum(gt_input, 0), 1)

        sample = {'arw': arw_input, 'rgb': rgb_input, 'gt': gt_input}

        if self.repeat_crop !=1:
            arw_stacks=[]
            rgb_stacks=[]
            gt_stacks=[]

            for i in range(self.repeat_crop):
                sample_patch = self.transform(sample)
                arw_stacks.append(sample_patch['arw'])
                rgb_stacks.append(sample_patch['rgb'])
                gt_stacks.append(sample_patch['gt'])
            return [torch.stack(arw_stacks), torch.stack(rgb_stacks)], torch.stack(gt_stacks)

        else:
            sample_patch = self.transform(sample)
            return [sample_patch['arw'], sample_patch['rgb']], sample_patch['gt']

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)
        im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out, W, H

