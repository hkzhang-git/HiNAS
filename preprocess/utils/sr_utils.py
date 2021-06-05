from glob import glob
from PIL import Image

import os


def sr_dict_build(args):
    dict_list = []
    for dataset in args.datasets:
        dict = []
        data_dir = '/'.join((args.data_root, args.task, dataset, 'HR'))

        im_list = glob(data_dir + '/*.jpg')
        if len(im_list) == 0:
            im_list = glob(data_dir + '/*.png')
        if len(im_list) == 0:
            im_list = glob(os.path.join(data_dir, '/*.bmp'))

        im_list.sort()

        for im_dir in im_list:
            if '\\' in im_dir:
                im_dir = im_dir.replace('\\', '/')
            # x2
            im_x2_dir = im_dir.replace('HR', 'LR/Bi_x2')
            with Image.open(im_x2_dir) as img:
                w_x2, h_x2 = img.width, img.height
            # x3
            im_x3_dir = im_dir.replace('HR', 'LR/Bi_x3')
            with Image.open(im_x3_dir) as img:
                w_x3, h_x3 = img.width, img.height
            # x4
            im_x4_dir = im_dir.replace('HR', 'LR/Bi_x4')
            with Image.open(im_x4_dir) as img:
                w_x4, h_x4 = img.width, img.height
            # x8
            im_x8_dir = im_dir.replace('HR', 'LR/Bi_x8')
            with Image.open(im_x8_dir) as img:
                w_x8, h_x8 = img.width, img.height

            sample_info = {
                'gt_path': '/'.join(im_dir.split('/')[-3:]),
                'x2_path': '/'.join(im_x2_dir.split('/')[-4:]),
                'x2_size': [int(w_x2), int(h_x2)],
                'x3_path': '/'.join(im_x3_dir.split('/')[-4:]),
                'x3_size': [int(w_x3), int(h_x3)],
                'x4_path': '/'.join(im_x4_dir.split('/')[-4:]),
                'x4_size': [int(w_x4), int(h_x4)],
                'x8_path': '/'.join(im_x8_dir.split('/')[-4:]),
                'x8_size': [int(w_x8), int(h_x8)],
            }
            dict.append(sample_info)
        dict_list.append(dict)

    return dict_list




