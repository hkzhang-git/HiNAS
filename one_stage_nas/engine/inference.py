import os
import logging
import numpy as np
import torch
import rawpy
import random
import matplotlib.pyplot as plt
from PIL import Image
from one_stage_nas.utils.evaluation_metrics import SSIM, PSNR
from one_stage_nas.data import build_transforms
from one_stage_nas.utils.misc import mkdir


def joint_patches(output_buffer, w, h, channel=3):
    if channel==3:
        count_matrix = np.zeros((int(h), int(w), 3), dtype=np.float32)
        im_result = torch.from_numpy(np.zeros((int(h), int(w), 3), dtype=np.float32))
        gt_result = torch.from_numpy(np.zeros((int(h), int(w), 3), dtype=np.float32))
    elif channel==1:
        count_matrix = np.zeros((int(h), int(w), 1), dtype=np.float32)
        im_result = torch.from_numpy(np.zeros((int(h), int(w), 1), dtype=np.float32))
        gt_result = torch.from_numpy(np.zeros((int(h), int(w), 1), dtype=np.float32))

    for item in output_buffer:
        im_patch = item['im_patch']
        gt_patch = item['gt_patch']
        crop_position = item['crop_position']
        w0, w1, h0, h1 = int(crop_position[0]), int(crop_position[1]), int(crop_position[2]), int(crop_position[3])

        im_result[h0:h1, w0:w1] = im_result[h0:h1, w0:w1] + im_patch.transpose(0, 2).transpose(0, 1).contiguous()
        gt_result[h0:h1, w0:w1] = gt_result[h0:h1, w0:w1] + gt_patch.transpose(0, 2).transpose(0, 1).contiguous()
        count_matrix[h0:h1, w0:w1] = count_matrix[h0:h1, w0:w1] + 1.0
    return im_result / torch.from_numpy(count_matrix), gt_result / torch.from_numpy(count_matrix)


def crop(crop_size, w, h):
    # slide_step = crop_size - crop_size // 4
    slide_step = crop_size
    # slide_step = crop_size
    x1 = list(range(0, w-crop_size, slide_step))
    x1.append(w-crop_size)
    y1 = list(range(0, h-crop_size, slide_step))
    y1.append(h-crop_size)

    x2 = [x+crop_size for x in x1]
    y2 = [y+crop_size for y in y1]

    return x1, x2, y1, y2


def truncated(input_tensor, max_l=1.0, min_l=0.0):
    input_tensor[input_tensor>max_l] = max_l
    input_tensor[input_tensor<min_l] = min_l

    return input_tensor


def tensor2img(input, output, target):
    b, c, h, w = target.shape

    input_img = []
    output_img = []
    target_img = []

    for i in range(b):
        input_img.append(input[i])
        output_img.append(output[i])
        target_img.append(target[i])

    return torch.cat(input_img, 1), torch.cat(output_img, 1), torch.cat(target_img, 1)


# inference related functions for denosing
def dn_inference(model, test_list, cfg, show_img=False, tag='search'):
    logger = logging.getLogger("one_stage_nas.inference")
    print('load test set')

    crop_size = cfg.DATASET.CROP_SIZE
    data_root = cfg.DATASET.DATA_ROOT

    test_dict = []
    for im_info in test_list:

        w, h = im_info['width'], im_info['height']
        im_id = im_info['path'].split('/')[-1]

        assert w >= crop_size and h >= crop_size
        x1, x2, y1, y2 = crop(crop_size, int(w), int(h))

        for x_start, x_end in zip(x1, x2):
            for y_start, y_end in zip(y1, y2):
                sample_info = {
                    'path': os.path.join(data_root, cfg.DATASET.TASK, '/'.join(im_info['path'].split('/')[-3:])),
                    'im_id': im_id,
                    'width': w,
                    'height': h,
                    'x1': x_start,
                    'x2': x_end,
                    'y1': y_start,
                    'y2': y_end
                }
                test_dict.append(sample_info)

    print('evaluation')
    transforms = build_transforms(task='dn', tag='test', sigma=cfg.DATALOADER.SIGMA)

    model.eval()
    metric_SSIM = SSIM(window_size=11, channel=cfg.MODEL.IN_CHANNEL, is_cuda=True)
    metric_PSNR = PSNR()

    batch_size = cfg.DATALOADER.BATCH_SIZE_TEST

    if tag == 'search':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                               '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                               format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                      cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES, cfg.MODEL.PRIMITIVES),
                                    'search/img_result'))

    elif tag == 'train':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                                    format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                           cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES, cfg.MODEL.PRIMITIVES),
                                    'train_noise_{}/img_result'.format(cfg.DATALOADER.SIGMA)))

    mkdir(result_save_dir)

    with torch.no_grad():
        previous_im_id = ''
        current_im_id = ''
        previous_im_w = None
        previous_im_h = None
        output_buffer = []

        dict_len = len(test_dict)
        batch_index_end = 0

        show_id = np.random.randint(0, dict_len // batch_size-1, 2)

        input_imgs = []
        output_imgs = []
        target_imgs = []

        i = 0
        while batch_index_end < dict_len:

            batch_index_start = batch_index_end
            batch_index_end = min(batch_index_end + batch_size, dict_len)

            images = []
            targets = []
            im_id = []
            w, h = [], []
            x1, x2, y1, y2 = [], [], [], []

            for index in range(batch_index_start, batch_index_end):
                patch_info = test_dict[index]
                if patch_info['im_id'] != current_im_id:
                    sample_data = Image.open(patch_info['path'])
                    width = patch_info['width']
                    height = patch_info['height']
                    current_im_id = patch_info['im_id']

                p_x1, p_x2, p_y1, p_y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']

                image = sample_data.crop((p_x1, p_y1, p_x2, p_y2))
                if cfg.DATASET.TO_GRAY:
                    image = image.convert('L')
                target = image

                sample = {'image': image, 'target': target}
                sample = transforms(sample)

                images.append(sample['image'])
                targets.append(sample['target'])
                im_id.append(patch_info['im_id'])
                w.append(width)
                h.append(height)
                x1.append(p_x1)
                x2.append(p_x2)
                y1.append(p_y1)
                y2.append(p_y2)

            images = torch.stack(images)
            targets = torch.stack(targets)
            output = model(images)

            if show_img and i in show_id:
                input_img, output_img, target_img = tensor2img(images, output, targets)
                input_imgs.append(input_img)
                output_imgs.append(output_img.cpu())
                target_imgs.append(target_img)

            for j in range(images.size(0)):
                if not (i == 0 and j == 0) and im_id[j] != previous_im_id:
                    im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, cfg.MODEL.IN_CHANNEL)
                    im_result[im_result > 1.0] = 1.0
                    im_result[im_result < 0.0] = 0.0

                    metric_SSIM(im_result.cuda(), gt_result.cuda())
                    metric_PSNR(im_result, gt_result)
                    im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
                    im_PIL.save(os.path.join(result_save_dir, previous_im_id))
                    output_buffer = []

                previous_im_id = im_id[j]
                previous_im_w = w[j]
                previous_im_h = h[j]

                patch_info = {
                    'im_patch': output[j].cpu(),
                    'gt_patch': targets[j],
                    'crop_position': [x1[j], x2[j], y1[j], y2[j]]
                }
                output_buffer.append(patch_info)

            i += 1

        im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, cfg.MODEL.IN_CHANNEL)
        im_result[im_result > 1.0] = 1.0
        im_result[im_result < 0.0] = 0.0
        metric_SSIM(im_result.cuda(), gt_result.cuda())
        metric_PSNR(im_result, gt_result)
        im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
        im_PIL.save(os.path.join(result_save_dir, previous_im_id))

    ssim = metric_SSIM.metric_get()
    psnr = metric_PSNR.metric_get()

    logger.info(' Val: SSIM:{} PSNR:{}'.format(ssim, psnr))

    if show_img:
        return ssim, psnr, truncated(torch.cat(input_imgs, 2), 1.0, 0.0), \
               truncated(torch.cat(output_imgs, 2), 1.0, 0.0), \
               torch.cat(target_imgs, 2)
    else:
        return ssim, psnr


# inference related functions for seeing in the dark
def sid_joint_patches(output_buffer, w, h, channel=3):
    if channel==3:
        count_matrix = np.zeros((int(h), int(w), 3), dtype=np.float32)
        im_result = torch.from_numpy(np.zeros((int(h), int(w), 3), dtype=np.float32))
        gt_result = torch.from_numpy(np.zeros((int(h), int(w), 3), dtype=np.float32))
    elif channel==1:
        count_matrix = np.zeros((int(h), int(w), 1), dtype=np.float32)
        im_result = torch.from_numpy(np.zeros((int(h), int(w), 1), dtype=np.float32))
        gt_result = torch.from_numpy(np.zeros((int(h), int(w), 1), dtype=np.float32))

    for item in output_buffer:
        im_patch = item['im_patch']
        gt_patch = item['gt_patch']
        crop_position = item['crop_position']
        w0, w1, h0, h1 = int(crop_position[0]), int(crop_position[1]), int(crop_position[2]), int(crop_position[3])

        im_result[h0:h1, w0:w1] = im_result[h0:h1, w0:w1] + im_patch.transpose(0, 2).transpose(0, 1).contiguous()
        gt_result[h0:h1, w0:w1] = gt_result[h0:h1, w0:w1] + gt_patch.transpose(0, 2).transpose(0, 1).contiguous()
        count_matrix[h0:h1, w0:w1] = count_matrix[h0:h1, w0:w1] + 1.0
    return im_result / torch.from_numpy(count_matrix), gt_result / torch.from_numpy(count_matrix)


def pack_raw(raw):
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


def sid_inference(model, test_list, cfg, show_img=False, tag='search'):
    logger = logging.getLogger("one_stage_nas.inference")
    print('load test set')

    crop_size = cfg.DATASET.CROP_SIZE
    data_root = cfg.DATASET.DATA_ROOT

    # rearrange the testing sample list
    test_list_new=[]
    for item in test_list:
        sample_id = item['sample_id']
        raw_path_arr = item['raw_path']
        gt_path = item['gt_path']
        raw_exposure_arr = item['raw_exposure']
        gt_exposure = item['gt_exposure']
        width = item['width']
        height = item['height']
        for raw_path, raw_exposure in zip(raw_path_arr, raw_exposure_arr):
            sample = {
                'sample_id': sample_id + '-{}s'.format(raw_exposure),
                'raw_path': raw_path,
                'gt_path': gt_path,
                'ratio': min(gt_exposure/raw_exposure, 300),
                'width': width,
                'height': height
            }
            test_list_new.append(sample)

    test_dict = []
    for sample_info in test_list_new:

        w, h = sample_info['width'], sample_info['height']
        assert w >= crop_size and h >= crop_size
        x1, x2, y1, y2 = crop(crop_size, int(w/2), int(h/2))
        sample_id = sample_info['sample_id']
        raw_path = sample_info['raw_path']
        gt_path = sample_info['gt_path']
        ratio = sample_info['ratio']

        for x_start, x_end in zip(x1, x2):
            for y_start, y_end in zip(y1, y2):
                sample_info = {
                    'sample_id': sample_id,
                    'raw_path': '/'.join((data_root, 'sid', raw_path)),
                    'gt_path': '/'.join((data_root, 'sid', gt_path)),
                    'ratio': ratio,
                    'width': w,
                    'height': h,
                    'x1': x_start,
                    'x2': x_end,
                    'y1': y_start,
                    'y2': y_end
                }
                test_dict.append(sample_info)

    print('evaluation')
    transforms = build_transforms(task='sid', tag='test')

    model.eval()
    metric_SSIM = SSIM(window_size=11, channel=3, is_cuda=True)
    metric_PSNR = PSNR()

    batch_size = cfg.DATALOADER.BATCH_SIZE_TEST

    if tag == 'search':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                                    format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                           cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES,
                                           cfg.MODEL.PRIMITIVES), 'search/img_result'))

    elif tag == 'train':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                                    format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                           cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES,
                                           cfg.MODEL.PRIMITIVES), 'train/img_result'))
    mkdir(result_save_dir)

    with torch.no_grad():
        previous_im_id = ''
        current_im_id = ''
        previous_im_w = None
        previous_im_h = None
        output_buffer = []

        dict_len = len(test_dict)
        batch_index_end = 0

        show_id = np.random.randint(0, dict_len // batch_size-1, 2)

        input_imgs = []
        output_imgs = []
        target_imgs = []

        i = 0
        while batch_index_end < dict_len:

            batch_index_start = batch_index_end
            batch_index_end = min(batch_index_end + batch_size, dict_len)

            arws = []
            rgbs = []
            gts = []
            sample_id = []
            w, h = [], []
            x1,y1 = [], []

            for index in range(batch_index_start, batch_index_end):
                patch_info = test_dict[index]
                if patch_info['sample_id'] != current_im_id:
                    width = patch_info['width']
                    height = patch_info['height']
                    ratio = patch_info['ratio']

                    raw_input = rawpy.imread(patch_info['raw_path'])
                    gt_input = rawpy.imread(patch_info['gt_path'])

                    arw_input, t_w, t_h = pack_raw(raw_input)
                    rgb_input = raw_input.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True,
                                                      output_bps=16)
                    rgb_input = (rgb_input / 65535.0).astype(np.float32)
                    gt_input = gt_input.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True,
                                                    output_bps=16)
                    gt_input = (gt_input / 65535.0).astype(np.float32)

                    arw_input = np.minimum(np.maximum(arw_input * ratio, 0), 1)
                    rgb_input = np.minimum(np.maximum(rgb_input * ratio, 0), 1)
                    gt_input = np.minimum(np.maximum(gt_input, 0), 1)

                    # raw_input = np.fromfile(patch_info['raw_path'], dtype=np.uint16).astype(np.float32)
                    # raw_input = pack_raw(raw_input, patch_info['width'], patch_info['height'])
                    # raw_input = np.minimum(np.maximum(raw_input * ratio, 0), 1)
                    #
                    # gt_input = np.array(Image.open(patch_info['gt_path']).copy()).astype(np.float32)
                    # gt_input = gt_input / 255
                    # gt_input = np.minimum(np.maximum(gt_input, 0), 1)

                    current_im_id = patch_info['sample_id']

                p_x1, p_x2, p_y1, p_y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']

                arw = arw_input[p_y1:p_y2, p_x1:p_x2, :]
                rgb = rgb_input[p_y1*2:(p_y1+crop_size)*2, p_x1*2:(p_x1+crop_size)*2, :]
                gt = gt_input[p_y1*2:(p_y1+crop_size)*2, p_x1*2:(p_x1+crop_size)*2, :]

                sample = {'arw': arw, 'rgb': rgb, 'gt': gt}
                sample = transforms(sample)

                arws.append(sample['arw'])
                rgbs.append(sample['rgb'])
                gts.append(sample['gt'])
                sample_id.append(patch_info['sample_id'])
                w.append(width)
                h.append(height)
                x1.append(p_x1)
                y1.append(p_y1)

            arw_inputs = torch.stack(arws)
            rgb_inputs = torch.stack(rgbs)
            targets = torch.stack(gts)
            output = model([arw_inputs, rgb_inputs])

            if show_img and i in show_id:
                input_img, output_img, target_img = tensor2img(arw_inputs, output, targets)
                input_imgs.append(input_img[0:4])
                output_imgs.append(output_img.cpu())
                target_imgs.append(target_img)

            for j in range(arw_inputs.size(0)):
                if not (i == 0 and j == 0) and sample_id[j] != previous_im_id:
                    im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, 3)
                    im_result[im_result > 1.0] = 1.0
                    im_result[im_result < 0.0] = 0.0

                    metric_SSIM(im_result.cuda(), gt_result.cuda())
                    metric_PSNR(im_result, gt_result)
                    im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
                    im_PIL.save(os.path.join(result_save_dir, previous_im_id + '.jpg'))
                    # im_PIL = Image.fromarray(np.array(gt_result.squeeze() * 255, np.uint8))
                    # im_PIL.save(os.path.join(result_save_dir, previous_im_id + 'gt.jpg'))
                    output_buffer = []

                previous_im_id = sample_id[j]
                previous_im_w = w[j]
                previous_im_h = h[j]

                patch_info = {
                    'im_patch': output[j].cpu(),
                    'gt_patch': targets[j],
                    'crop_position': [x1[j]*2, (x1[j]+crop_size)*2, y1[j]*2, (y1[j]+crop_size)*2]
                }
                output_buffer.append(patch_info)

            i += 1

        im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, 3)
        im_result[im_result > 1.0] = 1.0
        im_result[im_result < 0.0] = 0.0
        metric_SSIM(im_result.cuda(), gt_result.cuda())
        metric_PSNR(im_result, gt_result)
        im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
        im_PIL.save(os.path.join(result_save_dir, previous_im_id + '.jpg'))

    ssim = metric_SSIM.metric_get()
    psnr = metric_PSNR.metric_get()

    logger.info(' Val: SSIM:{} PSNR:{}'.format(ssim, psnr))

    if show_img:
        return ssim, psnr, truncated(torch.cat(input_imgs, 2), 1.0, 0.0), \
               truncated(torch.cat(output_imgs, 2), 1.0, 0.0), \
               torch.cat(target_imgs, 2)
    else:
        return ssim, psnr


def sr_joint_patches(output_buffer, w, h, channel=3):
    if channel==3:
        count_matrix = np.zeros((int(h), int(w), 3), dtype=np.float32)
        im_result = torch.from_numpy(np.zeros((int(h), int(w), 3), dtype=np.float32))
        gt_result = torch.from_numpy(np.zeros((int(h), int(w), 3), dtype=np.float32))
    elif channel==1:
        count_matrix = np.zeros((int(h), int(w), 1), dtype=np.float32)
        im_result = torch.from_numpy(np.zeros((int(h), int(w), 1), dtype=np.float32))
        gt_result = torch.from_numpy(np.zeros((int(h), int(w), 1), dtype=np.float32))

    for item in output_buffer:
        im_patch = item['im_patch']
        gt_patch = item['gt_patch']
        crop_position = item['crop_position']
        w0, w1, h0, h1 = int(crop_position[0]), int(crop_position[1]), int(crop_position[2]), int(crop_position[3])

        im_result[h0:h1, w0:w1] = im_result[h0:h1, w0:w1] + im_patch.transpose(0, 2).transpose(0, 1).contiguous()
        gt_result[h0:h1, w0:w1] = gt_result[h0:h1, w0:w1] + gt_patch.transpose(0, 2).transpose(0, 1).contiguous()
        count_matrix[h0:h1, w0:w1] = count_matrix[h0:h1, w0:w1] + 1.0
    return im_result / torch.from_numpy(count_matrix), gt_result / torch.from_numpy(count_matrix)


def sr_inference(model, test_list, cfg, show_img=False, tag='search'):
    logger = logging.getLogger("one_stage_nas.inference")
    print('load test set')

    crop_size = cfg.DATASET.CROP_SIZE
    data_root = cfg.DATASET.DATA_ROOT
    s_factor = cfg.DATALOADER.S_FACTOR
    # rearrange the testing sample list
    test_list_new=[]
    for item in test_list:
        sample_id = item['gt_path'].split('/')[-1]
        hr_path = item['gt_path']
        lr_path = item['x{}_path'.format(s_factor)]
        width, height = item['x{}_size'.format(s_factor)]

        sample = {
            'sample_id': sample_id,
            'hr_path': hr_path,
            'lr_path': lr_path,
            'width': width,
            'height': height
        }
        test_list_new.append(sample)

    test_dict = []
    for sample_info in test_list_new:

        w, h = sample_info['width'], sample_info['height']
        assert w >= crop_size and h >= crop_size
        x1, x2, y1, y2 = crop(crop_size, w, h)
        sample_id = sample_info['sample_id']
        hr_path = sample_info['hr_path']
        lr_path = sample_info['lr_path']

        for x_start, x_end in zip(x1, x2):
            for y_start, y_end in zip(y1, y2):
                sample_info = {
                    'sample_id': sample_id,
                    'hr_path': '/'.join((data_root, 'sr', hr_path)),
                    'lr_path': '/'.join((data_root, 'sr', lr_path)),
                    'width': w,
                    'height': h,
                    'x1': x_start,
                    'x2': x_end,
                    'y1': y_start,
                    'y2': y_end
                }
                test_dict.append(sample_info)

    print('evaluation')
    transforms = build_transforms(task='sr', tag='test')

    model.eval()
    metric_SSIM = SSIM(window_size=11, channel=3, is_cuda=True)
    metric_PSNR = PSNR()

    batch_size = cfg.DATALOADER.BATCH_SIZE_TEST

    if tag == 'search':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                                    format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                           cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES,
                                           cfg.MODEL.PRIMITIVES), 'search/img_result'))

    elif tag == 'train':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                                    format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                           cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES,
                                           cfg.MODEL.PRIMITIVES), 'train_x{}/img_result'.format(s_factor)))
    mkdir(result_save_dir)

    with torch.no_grad():
        previous_im_id = ''
        current_im_id = ''
        previous_im_w = None
        previous_im_h = None
        output_buffer = []

        dict_len = len(test_dict)
        batch_index_end = 0

        show_id = np.random.randint(0, dict_len // batch_size-1, 2)

        input_imgs = []
        output_imgs = []
        target_imgs = []

        i = 0
        while batch_index_end < dict_len:

            batch_index_start = batch_index_end
            batch_index_end = min(batch_index_end + batch_size, dict_len)

            images = []
            targets= []

            sample_id = []
            w, h = [], []
            x1, y1 = [], []

            for index in range(batch_index_start, batch_index_end):
                patch_info = test_dict[index]
                if patch_info['sample_id'] != current_im_id:
                    width = patch_info['width']
                    height = patch_info['height']

                    hr_im = Image.open(patch_info['hr_path']).crop((0, 0, width*s_factor, height*s_factor))
                    lr_im = Image.open(patch_info['lr_path'])

                    current_im_id = patch_info['sample_id']

                p_x1, p_x2, p_y1, p_y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']

                image = lr_im.crop((p_x1, p_y1, p_x2, p_y2))
                target = hr_im.crop((p_x1*s_factor, p_y1*s_factor, p_x2*s_factor, p_y2*s_factor))

                sample = {'image': image, 'target': target}
                sample = transforms(sample)

                images.append(sample['image'])
                targets.append(sample['target'])
                sample_id.append(patch_info['sample_id'])
                w.append(width)
                h.append(height)
                x1.append(p_x1)
                y1.append(p_y1)

            image_inputs = torch.stack(images)
            target_inputs = torch.stack(targets)
            output = model(image_inputs)

            if show_img and i in show_id:
                input_img, output_img, target_img = tensor2img(image_inputs, output, target_inputs)
                input_imgs.append(input_img)
                output_imgs.append(output_img.cpu())
                target_imgs.append(target_img)

            for j in range(image_inputs.size(0)):
                if not (i == 0 and j == 0) and sample_id[j] != previous_im_id:
                    im_result, gt_result = sr_joint_patches(output_buffer, previous_im_w*s_factor, previous_im_h*s_factor, 3)
                    im_result[im_result > 1.0] = 1.0
                    im_result[im_result < 0.0] = 0.0

                    metric_SSIM(im_result.cuda(), gt_result.cuda())
                    metric_PSNR(im_result, gt_result)
                    im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
                    im_PIL.save(os.path.join(result_save_dir, previous_im_id))
                    output_buffer = []

                previous_im_id = sample_id[j]
                previous_im_w = w[j]
                previous_im_h = h[j]

                patch_info = {
                    'im_patch': output[j].cpu(),
                    'gt_patch': targets[j],
                    'crop_position': [x1[j]*s_factor, (x1[j]+crop_size)*s_factor, y1[j]*s_factor, (y1[j]+crop_size)*s_factor]
                }
                output_buffer.append(patch_info)

            i += 1

        im_result, gt_result = joint_patches(output_buffer, previous_im_w*s_factor, previous_im_h*s_factor, 3)
        im_result[im_result > 1.0] = 1.0
        im_result[im_result < 0.0] = 0.0
        metric_SSIM(im_result.cuda(), gt_result.cuda())
        metric_PSNR(im_result, gt_result)
        im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
        im_PIL.save(os.path.join(result_save_dir, previous_im_id))

    ssim = metric_SSIM.metric_get()
    psnr = metric_PSNR.metric_get()

    logger.info(' Val: SSIM:{} PSNR:{}'.format(ssim, psnr))

    if show_img:
        return ssim, psnr, truncated(torch.cat(input_imgs, 2), 1.0, 0.0), \
               truncated(torch.cat(output_imgs, 2), 1.0, 0.0), \
               torch.cat(target_imgs, 2)
    else:
        return ssim, psnr