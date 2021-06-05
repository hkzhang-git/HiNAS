from glob import glob
from PIL import Image
import json
import os
import rawpy
import numpy as np


# def sid_dict_build(args):
#     data_dir = args.data_root + '/' + args.task
#     train_dict=[]
#     test_dict=[]
#
#     # build train_dict
#     with open(os.path.join(data_dir, 'Sony_train_list.txt'), 'r') as f:
#         train_info_list = f.readlines()
#         for train_info in train_info_list:
#             info = train_info.split()
#             raw_info = info[0]
#             raw_path = data_dir + raw_info[1:]
#             raw_exposure = info[0].split('/')[-1].split('_')[-1][:-5]
#             gt_info = info[1]
#             gt_path = (data_dir + gt_info[1:])
#             gt_exposure = info[1].split('/')[-1].split('_')[-1][:-5]
#             assert os.path.exists(raw_path) and os.path.exists(gt_path)
#             device = '_'.join((info[2], info[3]))
#             sample_id = info[0].split('/')[-1][:-4] + '-' + '{}s'.format(gt_exposure)
#             sample_info = {
#                 'sample_id': sample_id,
#                 'raw_path': '/'.join(raw_path.split('/')[-3:]),
#                 'gt_path': '/'.join(gt_path.split('/')[-3:]),
#                 'raw_exposure': float(raw_exposure),
#                 'gt_exposure': float(gt_exposure),
#                 'device': device
#             }
#             train_dict.append(sample_info)
#
#     with open(os.path.join(data_dir, 'Sony_val_list.txt'), 'r') as f:
#         train_info_list = f.readlines()
#         for train_info in train_info_list:
#             info = train_info.split()
#             raw_info = info[0]
#             raw_path = data_dir + raw_info[1:]
#             raw_exposure = info[0].split('/')[-1].split('_')[-1][:-5]
#             gt_info = info[1]
#             gt_path = (data_dir + gt_info[1:])
#             gt_exposure = info[1].split('/')[-1].split('_')[-1][:-5]
#             assert os.path.exists(raw_path) and os.path.exists(gt_path)
#             device = '_'.join((info[2], info[3]))
#             sample_id = info[0].split('/')[-1][:-4] + '-' + '{}s'.format(gt_exposure)
#             sample_info = {
#                 'sample_id': sample_id,
#                 'raw_path': '/'.join(raw_path.split('/')[-3:]),
#                 'gt_path': '/'.join(gt_path.split('/')[-3:]),
#                 'raw_exposure': float(raw_exposure),
#                 'gt_exposure': float(gt_exposure),
#                 'device': device
#             }
#             train_dict.append(sample_info)
#
#     # build test_dict
#     with open(os.path.join(data_dir, 'Sony_test_list.txt'), 'r') as f:
#         test_info_list = f.readlines()
#         for test_info in test_info_list:
#             info = test_info.split()
#             raw_info = info[0]
#             raw_path = data_dir + raw_info[1:]
#             raw_exposure = info[0].split('/')[-1].split('_')[-1][:-5]
#             gt_info = info[1]
#             gt_path = (data_dir + gt_info[1:])
#             gt_exposure = info[1].split('/')[-1].split('_')[-1][:-5]
#             assert os.path.exists(raw_path) and os.path.exists(gt_path)
#             device = '_'.join((info[2], info[3]))
#             sample_id = info[0].split('/')[-1][:-4] + '-' + '{}s'.format(gt_exposure)
#             sample_info = {
#                 'sample_id': sample_id,
#                 'raw_path': '/'.join(raw_path.split('/')[-3:]),
#                 'gt_path': '/'.join(gt_path.split('/')[-3:]),
#                 'raw_exposure': float(raw_exposure),
#                 'gt_exposure': float(gt_exposure),
#                 'device': device
#             }
#             test_dict.append(sample_info)
#
#     return train_dict, test_dict


def sid_dict_build(args):
    data_dir = args.data_root + '/' + args.task
    train_dict=[]
    test_dict=[]

    # build train_dict
    with open(os.path.join(data_dir, 'Sony_train_list.txt'), 'r') as f:
        train_info_list = f.readlines()
        raw_info_list = np.array([info.split()[0] for info in train_info_list])
        gt_info_list = [info.split()[1] for info in train_info_list]
        gt_set = list(set(gt_info_list))
        gt_set.sort()
        gt_info_list = np.array(gt_info_list)

        for gt_info in gt_set:
            gt_arw = rawpy.imread('/'.join((data_dir, gt_info)))
            width, height = gt_arw.sizes.iwidth, gt_arw.sizes.iheight
            raw_info_set = list(raw_info_list[gt_info_list==gt_info])
            sample_id = gt_info.split('/')[-1].split('.')[0]
            gt_exposure = gt_info.split('/')[-1].split('_')[-1][:-5]
            sample_info = {
                'sample_id': sample_id,
                'raw_path': [],
                'gt_path': gt_info[2:],
                'raw_exposure': [],
                'gt_exposure': float(gt_exposure),
                'width': width,
                'height': height,
            }
            for raw_info in raw_info_set:
                raw_path = raw_info[2:]
                raw_exposure = raw_info.split('/')[-1].split('_')[-1][:-5]
                sample_info['raw_path'].append(raw_path)
                sample_info['raw_exposure'].append(float(raw_exposure))

            train_dict.append(sample_info)

    with open(os.path.join(data_dir, 'Sony_val_list.txt'), 'r') as f:
        train_info_list = f.readlines()
        raw_info_list = np.array([info.split()[0] for info in train_info_list])
        gt_info_list = [info.split()[1] for info in train_info_list]
        gt_set = list(set(gt_info_list))
        gt_set.sort()
        gt_info_list = np.array(gt_info_list)

        for gt_info in gt_set:
            gt_arw = rawpy.imread('/'.join((data_dir, gt_info)))
            width, height = gt_arw.sizes.iwidth, gt_arw.sizes.iheight
            raw_info_set = list(raw_info_list[gt_info_list == gt_info])
            sample_id = gt_info.split('/')[-1].split('.')[0]
            gt_exposure = gt_info.split('/')[-1].split('_')[-1][:-5]
            sample_info = {
                'sample_id': sample_id,
                'raw_path': [],
                'gt_path': gt_info[2:],
                'raw_exposure': [],
                'gt_exposure': float(gt_exposure),
                'width': width,
                'height': height,
            }
            for raw_info in raw_info_set:
                raw_path = raw_info[2:]
                raw_exposure = raw_info.split('/')[-1].split('_')[-1][:-5]
                sample_info['raw_path'].append(raw_path)
                sample_info['raw_exposure'].append(float(raw_exposure))

            train_dict.append(sample_info)

    # build test_dict
    with open(os.path.join(data_dir, 'Sony_test_list.txt'), 'r') as f:
        test_info_list = f.readlines()
        for test_info in test_info_list:
            info = test_info.split()
            raw_info = info[0]
            raw_path = data_dir + raw_info[1:]
            raw_exposure = info[0].split('/')[-1].split('_')[-1][:-5]
            gt_info = info[1]
            gt_path = (data_dir + gt_info[1:])
            gt_exposure = info[1].split('/')[-1].split('_')[-1][:-5]
            assert os.path.exists(raw_path) and os.path.exists(gt_path)
            device = '_'.join((info[2], info[3]))
            sample_id = info[0].split('/')[-1][:-4] + '-' + '{}s'.format(gt_exposure)
            gt_arw = rawpy.imread(gt_path)
            width, height = gt_arw.sizes.iwidth, gt_arw.sizes.iheight
            sample_info = {
                'sample_id': sample_id,
                'raw_path': '/'.join(raw_path.split('/')[-3:]),
                'gt_path': '/'.join(gt_path.split('/')[-3:]),
                'raw_exposure': float(raw_exposure),
                'gt_exposure': float(gt_exposure),
                'device': device,
                'width': width,
                'height': height,
            }
            test_dict.append(sample_info)

    return train_dict, test_dict

