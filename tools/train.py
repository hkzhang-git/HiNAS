"""
Searching script
"""

import argparse
from tensorboardX import SummaryWriter

import torch
import os
import sys
sys.path.append('..')
from one_stage_nas.config import cfg
from one_stage_nas.data import build_dataset
from one_stage_nas.solver import make_lr_scheduler
from one_stage_nas.solver import make_optimizer
from one_stage_nas.engine.trainer import do_train
from one_stage_nas.modeling.architectures import build_model
from one_stage_nas.utils.checkpoint import Checkpointer
from one_stage_nas.utils.logger import setup_logger
from one_stage_nas.utils.misc import mkdir
from one_stage_nas.utils.visualize import visualize


def train(cfg, output_dir):
    model = build_model(cfg)

    # visualize
    visual_dir = output_dir + '/arch'
    geno_cell, geno_path = model.genotype()
    visualize(geno_cell, geno_path, visual_dir, cfg.SEARCH.TIE_CELL)

    # just use data parallel
    model = torch.nn.DataParallel(model).cuda()

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    checkpointer = Checkpointer(
        model, optimizer, scheduler, output_dir + '/models', save_to_disk=True)

    train_loader, val_list = build_dataset(cfg)

    arguments = {}
    arguments["iteration"] = 0
    arguments["genotype"] = model.module.genotype()

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    val_period = cfg.SOLVER.VALIDATE_PERIOD
    max_iter = cfg.SOLVER.TRAIN.MAX_ITER

    writer = SummaryWriter(logdir=output_dir + '/log', comment=cfg.DATASET.TASK + '_' + cfg.DATASET.DATA_NAME)

    do_train(
        model,
        train_loader,
        val_list,
        max_iter,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpoint_period,
        arguments,
        writer,
        cfg
    )


def main():
    parser = argparse.ArgumentParser(description="One-stage NAS Training")
    parser.add_argument(
        "--config-file",
        default="../configs/sr/DIV2K_2c3n/03_x4_train_CR.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--device",
        default='4',
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.DATASET.TASK in ['dn']:
        output_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                               '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                               format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                      cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES, cfg.MODEL.PRIMITIVES),
                               'train_noise_{}'.format(cfg.DATALOADER.SIGMA)))
    elif cfg.DATASET.TASK in ['sid']:
        output_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                               '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                               format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                      cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES, cfg.MODEL.PRIMITIVES),
                               'train'))
    elif cfg.DATASET.TASK in ['sr']:
        output_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                               '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                               format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                      cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES, cfg.MODEL.PRIMITIVES),
                               'train_x{}'.format(cfg.DATALOADER.S_FACTOR)))

    mkdir(output_dir+'/models')

    logger = setup_logger("one_stage_nas", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, output_dir)


if __name__ == "__main__":
    main()
