import logging
import time
import datetime
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from one_stage_nas.utils.metric_logger import MetricLogger
from one_stage_nas.utils.comm import reduce_loss_dict, compute_params
from .inference import denoise_inference_joint
from one_stage_nas.utils.evaluation_metrics import SSIM, PSNR


def do_train(
        model,
        train_loader_list,
        val_set_list,
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

    inference = denoise_inference_joint

    best_val = 0
    model.train()

    data_iter_list = []
    for train_loader in train_loader_list:
        data_iter_list.append(iter(train_loader))

    meters = MetricLogger(delimiter="  ")
    metric_SSIM = SSIM(window_size=11, channel=cfg.MODEL.IN_CHANNEL, is_cuda=True)
    metric_PSNR = PSNR()

    datasets_weight = np.array(cfg.DATASET.TRAIN_DATASETS_WEIGHT)
    weights = [datasets_weight[:i+1].sum() for i in range(len(datasets_weight))]
    end = time.time()
    for iteration in range(start_iter, max_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        random_id = random_choose(weights)
        try:
            images, targets = next(data_iter_list[random_id])
        except StopIteration:
            data_iter_list = []
            for train_loader in train_loader_list:
                data_iter_list.append(iter(train_loader))
            images, targets = next(data_iter_list[random_id])
        data_time = time.time() - end

        pred, loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values()).mean()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3)
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

        if iteration % val_period == 0:
            train_ssim, train_psnr = metric_SSIM.metric_get(), metric_PSNR.metric_get()
            metric_SSIM.reset()
            metric_PSNR.reset()

            ssim, psnr, input_img, output_img, target_img = inference(model, val_set_list, cfg, show_img=True, tag='train')
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
            # writer.add_scalars('SSIM', {'train_ssim': train_ssim}, iteration)
            # writer.add_scalars('PSNR', {'train_psnr': train_psnr}, iteration)

        if iteration % val_period == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {}".format(total_time_str))

    writer.close()


def random_choose(weights):
    random_id = random.randint(1, weights[-1])
    for id, region in enumerate(weights):
        if random_id<=region:
            return id


