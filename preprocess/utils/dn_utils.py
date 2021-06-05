from glob import glob
from PIL import Image
import json
import os


def denoise_dict_build(args):
    dict_list = []
    for dataset in args.datasets:
        if dataset in ['BSD500_300', 'BSD500_200', 'Urben100', 'set14']:
            dict = []
            data_dir = os.path.join(args.data_root, args.task, dataset)

            im_list = glob(os.path.join(data_dir, '*.jpg'))
            if len(im_list) == 0:
                im_list = glob(os.path.join(data_dir, '*.png'))
            if len(im_list) == 0:
                im_list = glob(os.path.join(data_dir, '*.bmp'))

            im_list.sort()

            for im_dir in im_list:
                if '\\' in im_dir:
                    im_dir=im_dir.replace('\\', '/')
                with Image.open(im_dir) as img:
                    w, h = img.width, img.height

                sample_info = {
                    'path': '/'.join(im_dir.split('/')[-2:]),
                    'width': int(w),
                    'height': int(h)
                }
                dict.append(sample_info)
            dict_list.append(dict)

    return dict_list


def json_save(save_path, dict_file):
    with open(save_path, 'w') as f:
        json.dump(dict_file, f)


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

