"""
Searching script
"""

import argparse
import os
import json
import torch
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')
from one_stage_nas.config import cfg
from one_stage_nas.data import build_transforms
from one_stage_nas.utils.misc import mkdir
from one_stage_nas.modeling.architectures import build_model
from PIL import Image
from one_stage_nas.utils.evaluation_metrics import SSIM, PSNR

import time

def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


def evaluation(cfg, s_factor, dataset):
    print('load test set')
    dataset_json_dir = '/'.join((cfg.DATALOADER.DATA_LIST_DIR, cfg.DATASET.TASK, '{}.json'.format(dataset)))
    data_dict = json_loader(dataset_json_dir)

    data_root = cfg.DATASET.DATA_ROOT
    s_factor = cfg.DATALOADER.S_FACTOR
    # rearrange the testing sample list

    print('model build')

    trained_model_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                  '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                                  format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                         cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES,
                                         cfg.MODEL.PRIMITIVES),
                                  'train_x{}/models/model_best.pth'.format(s_factor)))

    if not os.path.exists(trained_model_dir):
        print('trained_model does not exist')
        return None, None
    model = build_model(cfg)
    model = torch.nn.DataParallel(model).cuda()

    model_state_dict = torch.load(trained_model_dir).pop("model")
    try:
        model.load_state_dict(model_state_dict)
    except:
        model.module.load_state_dict(model_state_dict)

    print('dataset {} evaluation...'.format(dataset))

    transforms = build_transforms(task='sr', tag='test')

    # as we record the PSNR and SSIM on Y channel, here the number of input channel is set to 1
    model.eval()
    metric_SSIM = SSIM(window_size=11, channel=1, is_cuda=True)
    metric_PSNR = PSNR()

    result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                                format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                       cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES,
                                       cfg.MODEL.PRIMITIVES),
                                'eval_x{}/'.format(s_factor), dataset))

    mkdir(result_save_dir)

    with torch.no_grad():

        for item in data_dict:
            sample_id = item['gt_path'].split('/')[-1]
            hr_path = item['gt_path']
            lr_path = item['x{}_path'.format(s_factor)]
            width, height = item['x{}_size'.format(s_factor)]

            hr_im = Image.open('/'.join((data_root, 'sr', hr_path))).crop((0, 0, width * s_factor, height * s_factor))
            lr_im = Image.open('/'.join((data_root, 'sr', lr_path)))

            sample = {'image': lr_im, 'target': hr_im}
            sample = transforms(sample)

            image, target = sample['image'], sample['target']

            output = model(image.unsqueeze(dim=0))

            im_result = output.squeeze().transpose(0, 2).transpose(0, 1)
            gt_result = target.transpose(0, 2).transpose(0, 1)

            im_result[im_result > 1.0] = 1.0
            im_result[im_result < 0.0] = 0.0

            im_result_Y = (im_result[:, :, 0] * 24.966 +
                           im_result[:, :, 1] * 128.553 +
                           im_result[:, :, 2] * 65.481 +
                           16.0) / 255.0
            gt_result_Y = (gt_result[:, :, 0] * 24.966 +
                           gt_result[:, :, 1] * 128.553 +
                           gt_result[:, :, 2] * 65.481 +
                           16.0) / 255.0

            im_result_Y = im_result_Y.unsqueeze(dim=2)
            gt_result_Y = gt_result_Y.unsqueeze(dim=2)

            metric_SSIM(im_result_Y, gt_result_Y.cuda())
            metric_PSNR(im_result_Y, gt_result_Y.cuda())
            im_PIL = Image.fromarray(np.array(im_result.cpu().squeeze() * 255, np.uint8))
            im_PIL.save(os.path.join(result_save_dir, sample_id))

    ssim = metric_SSIM.metric_get()
    psnr = metric_PSNR.metric_get()

    print('dataset:{} ssim:{}, psnr:{}'.format(dataset, ssim, psnr))
    with open(os.path.join(result_save_dir, 'evaluation_result.txt'), 'w') as f:
        f.write('SSIM:{} PSNR:{}'.format(ssim, psnr))


def main():
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument(
        "--config-file",
        default="../configs/sr/DIV2K_2c3n/03_x4_infe_CR.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--device",
        default='0',
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    for dataset in cfg.DATASET.TEST_DATASETS:
        evaluation(cfg, cfg.DATALOADER.S_FACTOR, dataset)


if __name__ == "__main__":
    main()
