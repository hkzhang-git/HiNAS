import logging
import time
import datetime

import torch
import matplotlib.pyplot as plt

from one_stage_nas.utils.metric_logger import MetricLogger
from one_stage_nas.utils.comm import reduce_loss_dict, compute_params
from .inference import dn_inference, sid_inference, sr_inference
from one_stage_nas.utils.evaluation_metrics import SSIM, PSNR


def do_train(
        model,
        train_loader,
        val_list,
        max_iter,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpointer_period,
        arguments,
        writer,
        cfg):
    """
    num_classes (int): number of classes. Required by computing mIoU.
    """
    logger = logging.getLogger("one_stage_nas.trainer")
    logger.info("Model Params: {:.2f}M".format(compute_params(model) / 1024 / 1024))

    logger.info("Start training")

    start_iter = arguments["iteration"]
    start_training_time = time.time()

    if cfg.DATASET.TASK == 'dn':
        inference = dn_inference
    elif cfg.DATASET.TASK == 'sid':
        inference = sid_inference
    elif cfg.DATASET.TASK == 'sr':
        inference = sr_inference


    best_val = 0
    model.train()
    data_iter = iter(train_loader)

    meters = MetricLogger(delimiter="  ")
    if cfg.DATASET.TASK in ['sid']:
        metric_SSIM = SSIM(window_size=11, channel=3, is_cuda=True)
    else:
        metric_SSIM = SSIM(window_size=11, channel=cfg.MODEL.IN_CHANNEL, is_cuda=True)
    metric_PSNR = PSNR()
    repeat_crop = cfg.DATALOADER.R_CROP

    end = time.time()
    for iteration in range(start_iter, max_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        data_time = time.time() - end

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

        pred, loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values()).mean()

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_loss_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        pred[pred>1.0] = 1.0
        pred[pred<0.0] = 0.0

        targets = targets.cuda()

        metric_SSIM(pred.detach(), targets, transpose=False)
        metric_PSNR(pred.detach(), targets)

        if iteration % (val_period // 4) == 0:
            logger.info(
                meters.delimiter.join(
                ["eta: {eta}",
                 "iter: {iter}",
                 "{meters}",
                 "lr: {lr:.6f}",
                 "max_mem: {memory:.0f}"]).format(
                     eta=eta_string,
                     iter=iteration,
                     meters=str(meters),
                     lr=optimizer.param_groups[0]['lr'],
                     memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
            print(float(losses))

        if iteration % val_period == 0:
            train_ssim, train_psnr = metric_SSIM.metric_get(), metric_PSNR.metric_get()
            metric_SSIM.reset()
            metric_PSNR.reset()

            if iteration > int(max_iter*3/4):
                ssim, psnr, input_img, output_img, target_img = inference(model, val_list, cfg, show_img=True, tag='train')
                if best_val < (ssim + psnr/100):
                    best_val = (ssim + psnr/100)
                    checkpointer.save("model_best", **arguments)
                # set mode back to train
                model.train()
                writer.add_image('img/train/input', input_img, iteration)
                writer.add_image('img/train/output', output_img, iteration)
                writer.add_image('img/train/target', target_img, iteration)
                writer.add_scalars('SSIM', {'train_ssim': train_ssim, 'val_ssim': ssim}, iteration)
                writer.add_scalars('PSNR', {'train_psnr': train_psnr, 'val_psnr': psnr}, iteration)
            else:
                writer.add_scalars('SSIM', {'train_ssim': train_ssim}, iteration)
                writer.add_scalars('PSNR', {'train_psnr': train_psnr}, iteration)

        if iteration % val_period == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {}".format(total_time_str))

    writer.close()

