import os
import logging
import time
import datetime

import torch
import matplotlib.pyplot as plt

from one_stage_nas.utils.metric_logger import MetricLogger
from one_stage_nas.utils.comm import reduce_loss_dict
from one_stage_nas.utils.visualize import model_visualize
# from .inference import dn_inference, sid_inference, sr_inference
from .inference import dn_inference, sr_inference


def do_search(
        model,
        train_loaders,
        val_list,
        max_epoch,
        arch_start_epoch,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpointer_period,
        arguments,
        writer,
        cfg,
        visual_dir):
    """
    num_classes (int): number of classes. Required by computing mIoU.
    """
    logger = logging.getLogger("one_stage_nas.searcher")
    logger.info("Start searching")

    start_epoch = arguments["epoch"]
    start_training_time = time.time()

    if cfg.DATASET.TASK == 'dn':
        inference = dn_inference
    # elif cfg.DATASET.TASK == 'sid':
    #     inference = sid_inference
    elif cfg.DATASET.TASK == 'sr':
        inference = sr_inference

    best_val = 0
    for epoch in range(start_epoch, max_epoch):
        epoch = epoch + 1
        arguments["epoch"] = epoch

        scheduler.step()

        train(model, train_loaders, optimizer, epoch,
              train_arch=epoch > arch_start_epoch, repeat_crop=cfg.DATALOADER.R_CROP)
        if epoch > cfg.SEARCH.ARCH_START_EPOCH:
            save_dir = '/'.join((visual_dir, 'visualize', 'arch_epoch{}'.format(epoch)))
            model_visualize(model, save_dir, cfg.SEARCH.TIE_CELL)
        if epoch % val_period == 0 and epoch > 60:
        # if epoch % val_period == 0:
            ssim, psnr = inference(model, val_list, cfg)
            if best_val < (ssim + psnr/100):
                best_val = (ssim + psnr/100)
                checkpointer.save("model_best", **arguments)
            writer.add_scalars('Search_SSIM', { 'val_ssim': ssim}, epoch)
            writer.add_scalars('Search_PSNR', {'val_psnr': psnr}, epoch)
        if epoch % checkpointer_period == 0:
            checkpointer.save("model_{:03d}".format(epoch), **arguments)
        if epoch == max_epoch:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {}".format(total_time_str))


def train(model, data_loaders, optimizer, epoch,
          train_arch=False, repeat_crop=1):
    """
    Should add some stats and log to visualise the archs
    """
    data_loader_w = data_loaders[0]
    data_loader_a = data_loaders[1]
    optim_w = optimizer['optim_w']
    optim_a = optimizer['optim_a']

    logger = logging.getLogger("one_stage_nas.searcher")

    max_iter = len(data_loader_w)
    model.train()
    meters = MetricLogger(delimiter="  ")
    end = time.time()
    for iteration, (images, targets) in enumerate(data_loader_w):
        data_time = time.time() - end

        if train_arch:
            # print('start_train_arch')
            images_a, targets_a = next(iter(data_loader_a))

            if repeat_crop != 1:
                if isinstance(images_a, list):
                    ima0_sizes = images_a[0].shape
                    ima1_sizes = images_a[1].shape
                    images_a = [images_a[0].view(ima0_sizes[0] * ima0_sizes[1], ima0_sizes[2], ima0_sizes[3], ima0_sizes[4]),
                              images_a[1].view(ima1_sizes[0] * ima1_sizes[1], ima1_sizes[2], ima1_sizes[3], ima1_sizes[4])
                              ]
                else:
                    im_sizes = images_a.shape
                    images_a = images_a.view(im_sizes[0] * im_sizes[1], im_sizes[2], im_sizes[3], im_sizes[4])

                ta_sizes = targets_a.shape
                targets_a = targets.view(ta_sizes[0] * ta_sizes[1], ta_sizes[2], ta_sizes[3], ta_sizes[4])

            loss_dict = model(images_a, targets_a)
            losses = sum(loss for loss in loss_dict.values()).mean()

            optim_a.zero_grad()
            losses.backward()
            optim_a.step()

        if repeat_crop!=1:

            if isinstance(images, list):
                im0_sizes = images[0].shape
                im1_sizes = images[1].shape
                images = [images[0].view(im0_sizes[0]*im0_sizes[1], im0_sizes[2], im0_sizes[3], im0_sizes[4]),
                          images[1].view(im1_sizes[0]*im1_sizes[1], im1_sizes[2], im1_sizes[3], im1_sizes[4])
                          ]
            else:
                im_sizes = images.shape
                images = images.view(im_sizes[0] * im_sizes[1], im_sizes[2], im_sizes[3], im_sizes[4])

            ta_sizes = targets.shape
            targets = targets.view(ta_sizes[0]*ta_sizes[1], ta_sizes[2], ta_sizes[3], ta_sizes[4])

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values()).mean()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optim_w.zero_grad()
        losses.backward()
        optim_w.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 50 == 0:
            logger.info(
                meters.delimiter.join(
                ["eta: {eta}",
                 "iter: {epoch}/{iter}",
                 "{meters}",
                 "lr: {lr:.6f}",
                 "max_mem: {memory:.1f} G"]).format(
                     eta=eta_string,
                     epoch=epoch,
                     iter=iteration,
                     meters=str(meters),
                     lr=optim_w.param_groups[0]['lr'],
                     memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0))
