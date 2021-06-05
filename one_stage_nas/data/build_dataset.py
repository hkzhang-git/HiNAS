from .datasets.transforms import (RandomCrop, RandomMirror, RandomOverturn,
                                  RandomRotate, FourLRotate, Normalize,
                                  RandomRescaleCrop, ToTensor, NoiseToTensor,
                                  SID_RandomCrop, SID_RandomFlip, Ndarray2tensor,
                                  Sr_RandomCrop, Rescale, Compose)
from .datasets.tasks_dict import tasks_dict
import numpy as np
import torch
import json
import os


def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


def build_transforms(crop_size=None, task='dn', tag='train', sigma=[], s_factor=1):

    if task == 'dn':
        if tag == 'train':
            return Compose([
                RandomRescaleCrop(crop_size),
                FourLRotate(),
                RandomMirror(),
                NoiseToTensor(sigma),
                Rescale()
            ])
        elif tag == 'test':
            return Compose([
                NoiseToTensor(sigma),
                Rescale(),
            ])
    elif task == 'sid':
        if tag == 'train':
            return Compose([
                SID_RandomCrop(crop_size),
                SID_RandomFlip(),
                Ndarray2tensor(),
            ])
        elif tag == 'test':
            return Compose([
                Ndarray2tensor(),
            ])
    elif task == 'sr':
        if tag == 'train':
            return Compose([
                Sr_RandomCrop(crop_size, s_factor),
                FourLRotate(),
                RandomMirror(),
                ToTensor(),
                Rescale()
            ])
        elif tag == 'test':
            return Compose([
                ToTensor(),
                Rescale(),
            ])

def build_dataset(cfg):
    data_root = cfg.DATASET.DATA_ROOT
    data_name = cfg.DATASET.DATA_NAME
    task = cfg.DATASET.TASK

    if cfg.SEARCH.SEARCH_ON:
        crop_size = cfg.DATASET.CROP_SIZE
    else:
        crop_size = cfg.INPUT.CROP_SIZE_TRAIN

    data_list_dir = cfg.DATALOADER.DATA_LIST_DIR
    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.DATALOADER.BATCH_SIZE_TRAIN

    search_on = cfg.SEARCH.SEARCH_ON
    transform = build_transforms(crop_size, task, tag='train', sigma=cfg.DATALOADER.SIGMA, s_factor=cfg.DATALOADER.S_FACTOR)

    if task in ['dn', 'sr']:
        data_dict = json_loader('/'.join((data_list_dir, task, data_name + '.json')))
    elif task in ['sid']:
        data_dict = json_loader('/'.join((data_list_dir, task, data_name, 'train.json')))

    if search_on:
        num_samples = len(data_dict)
        val_split = int(np.floor(cfg.SEARCH.VAL_PORTION * num_samples))
        num_train = num_samples - val_split
        train_split = int(np.floor(cfg.SEARCH.PORTION * num_train))
        w_data_list = [data_dict[i] for i in range(train_split)]
        a_data_list = [data_dict[i] for i in range(train_split, num_train)]
        v_data_list = [data_dict[i] for i in range(num_train, num_samples)]

        dataset_w = tasks_dict[task]('/'.join((data_root, task)), w_data_list, transform,
                                     cfg.DATASET.LOAD_ALL, cfg.DATASET.TO_GRAY, cfg.DATALOADER.S_FACTOR, cfg.DATALOADER.R_CROP)
        dataset_a = tasks_dict[task]('/'.join((data_root, task)), a_data_list, transform,
                                     cfg.DATASET.LOAD_ALL, cfg.DATASET.TO_GRAY, cfg.DATALOADER.S_FACTOR, cfg.DATALOADER.R_CROP)

        data_loader_w = torch.utils.data.DataLoader(
            dataset_w,
            shuffle=True,
            batch_size=batch_size // cfg.DATALOADER.R_CROP,
            num_workers=min(num_workers, batch_size // cfg.DATALOADER.R_CROP),
            pin_memory=True)

        data_loader_a = torch.utils.data.DataLoader(
            dataset_a,
            shuffle=True,
            batch_size=batch_size // cfg.DATALOADER.R_CROP,
            num_workers=min(num_workers, batch_size // cfg.DATALOADER.R_CROP),
            pin_memory=True)

        return [data_loader_w, data_loader_a], v_data_list
    else:
        num_samples = len(data_dict)
        val_split = int(np.floor(cfg.SEARCH.VAL_PORTION * num_samples))
        num_train = num_samples - val_split

        t_data_list = [data_dict[i] for i in range(num_train)]
        v_data_list = [data_dict[i] for i in range(num_train, num_samples)]

        dataset_t = tasks_dict[task]('/'.join((data_root, task)), t_data_list, transform,
                                     cfg.DATASET.LOAD_ALL, cfg.DATASET.TO_GRAY, cfg.DATALOADER.S_FACTOR, cfg.DATALOADER.R_CROP)

        data_loader_t = torch.utils.data.DataLoader(
            dataset_t,
            shuffle=True,
            batch_size=batch_size // cfg.DATALOADER.R_CROP,
            num_workers=min(num_workers, batch_size // cfg.DATALOADER.R_CROP),
            pin_memory=True)

        return data_loader_t, v_data_list

