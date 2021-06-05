"""
Searching script
"""
import argparse
import torch
import os
import sys
sys.path.append('..')
from one_stage_nas.config import cfg
from one_stage_nas.data import build_dataset
from one_stage_nas.solver import make_lr_scheduler
from one_stage_nas.solver import make_optimizer
from one_stage_nas.engine.searcher import do_search
from one_stage_nas.modeling.architectures import build_model
from one_stage_nas.utils.checkpoint import Checkpointer
from one_stage_nas.utils.logger import setup_logger
from one_stage_nas.utils.misc import mkdir
from tensorboardX import SummaryWriter


def search(cfg, output_dir):

    # set random seed
    torch.manual_seed(cfg.SEARCH.R_SEED)
    torch.cuda.manual_seed(cfg.SEARCH.R_SEED)

    model = build_model(cfg)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    checkpointer = Checkpointer(
        model, optimizer, scheduler, output_dir + '/models', save_to_disk=True)

    train_loaders, val_dict = build_dataset(cfg)

    arguments = {}
    arguments["epoch"] = 0

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    # just use data parallel
    model = torch.nn.DataParallel(model).cuda()

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    val_period = cfg.SOLVER.VALIDATE_PERIOD
    max_epoch = cfg.SOLVER.MAX_EPOCH
    arch_start_epoch = cfg.SEARCH.ARCH_START_EPOCH

    writer = SummaryWriter(logdir=output_dir + '/log', comment=cfg.DATASET.TASK + '_' + cfg.DATASET.DATA_NAME)

    do_search(
        model,
        train_loaders,
        val_dict,
        max_epoch,
        arch_start_epoch,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpoint_period,
        arguments,
        writer,
        cfg,
        visual_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="neural architecture search for four different low-level tasks")
    parser.add_argument(
        "--config-file",
        default='../configs/dn/BSD500_3c4n/03_search_CR_R0.yaml',
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--device",
        default='3',
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

    output_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                           '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                           format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                  cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES, cfg.MODEL.PRIMITIVES),
                           'search'))
    mkdir(output_dir)
    mkdir(output_dir + '/models')
    logger = setup_logger("one_stage_nas", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    search(cfg, output_dir)


if __name__ == "__main__":
    main()
