"""
Discrete structure of Auto-DeepLab

Includes utils to convert continous Auto-DeepLab to discrete ones
"""

import os
import torch
from torch import nn
from torch.nn import functional as F

from one_stage_nas.darts.cell import FixCell
from .dn_supernet import Dn_supernet
from .common import conv3x3_bn, conv1x1_bn
from .decoders import build_decoder
from .loss import loss_dict


def get_genotype_from_adl(cfg):
    # create ADL model
    adl_cfg = cfg.clone()
    adl_cfg.defrost()

    adl_cfg.merge_from_list(['MODEL.META_ARCHITECTURE', 'AutoDeepLab',
                             'MODEL.FILTER_MULTIPLIER', 8,
                             'MODEL.AFFINE', True,
                             'SEARCH.SEARCH_ON', True])

    model = Dn_supernet(adl_cfg)
    # load weights
    SEARCH_RESULT_DIR = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                           '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                           format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                  cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES, cfg.MODEL.PRIMITIVES),
                                  'search/models/model_best.pth'))
    ckpt = torch.load(SEARCH_RESULT_DIR)
    restore = {k: v for k, v in ckpt['model'].items() if 'arch' in k}
    model.load_state_dict(restore, strict=False)
    return model.genotype()


class DeepLabScaler_Width(nn.Module):
    """Official implementation
    https://github.com/tensorflow/models/blob/master/research/deeplab/core/nas_cell.py#L90
    """
    def __init__(self, inp, C, activate_f='ReLU'):
        super(DeepLabScaler_Width, self).__init__()
        self.activate_f = activate_f
        self.conv = conv1x1_bn(inp, C, 1, activate_f=None)

    def forward(self, hidden_state):
        if self.activate_f.lower() == 'relu':
            return self.conv(F.relu(hidden_state))
        elif self.activate_f.lower() in ['leaky', 'prelu']:
            return self.conv(F.leaky_relu(hidden_state, negative_slope=0.2))
        elif self.activate_f.lower() == 'sine':
            return self.conv(torch.sin(hidden_state))


class Dn_compnet(nn.Module):
    def __init__(self, cfg):
        super(Dn_compnet, self).__init__()

        # load genotype
        if len(cfg.DATASET.TRAIN_DATASETS) == 0:
            geno_file = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                   '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                                   format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                          cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES, cfg.MODEL.PRIMITIVES),
                                  'search/models/model_best.geno'))

        else:
            geno_file = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                  '{}/Outline-{}c{}n_TC-{}_ASPP-{}_Res-{}_Prim-{}'.
                                  format(cfg.DATASET.DATA_NAME, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                         cfg.SEARCH.TIE_CELL, cfg.MODEL.USE_ASPP, cfg.MODEL.USE_RES, cfg.MODEL.PRIMITIVES),
                                  'search/models/model_best.geno'))

        if os.path.exists(geno_file):
            print("Loading genotype from {}".format(geno_file))
            genotype = torch.load(geno_file, map_location=torch.device("cpu"))
        else:
            genotype = get_genotype_from_adl(cfg)
            print("Saving genotype to {}".format(geno_file))
            torch.save(genotype, geno_file)

        geno_cell, geno_path = genotype

        self.genotpe = genotype

        if 0 in geno_path:
            self.endpoint = (len(geno_path) - 1) - list(reversed(geno_path)).index(0)
            if self.endpoint == (len(geno_path) -1):
                self.endpoint = None
        else:
            self.endpoint = None

        # basic configs
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.use_res = cfg.MODEL.USE_RES
        self.f = cfg.MODEL.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.num_strides = cfg.MODEL.NUM_STRIDES
        self.ws_factors = cfg.MODEL.WS_FACTORS
        self.in_channel = cfg.MODEL.IN_CHANNEL
        self.stem1 = conv3x3_bn(self.in_channel, 64, 1, activate_f=self.activate_f)
        self.stem2 = conv3x3_bn(64, 64, 1, activate_f=None)
        self.reduce = conv3x3_bn(64, self.f*self.num_blocks, 1, affine=False, activate_f=None)

        # create cells
        self.cells = nn.ModuleList()
        self.scalers = nn.ModuleList()
        if cfg.SEARCH.TIE_CELL:
            geno_cell = [geno_cell] * self.num_layers

        DeepLabScaler = DeepLabScaler_Width

        h_0 = 0  # prev hidden index
        h_1 = -1  # prev prev hidden index
        for layer, (geno, h_ind) in enumerate(zip(geno_cell, geno_path), 1):
            stride = self.ws_factors[h_ind]
            h = self.ws_factors[h_ind]
            self.cells.append(FixCell(geno, int(self.f * stride)))
            # scalers
            if layer == 1:
                inp0 = 64
                inp1 = 64
            elif layer == 2:
                inp0 = int(h_0 * self.f * self.num_blocks)
                inp1 = 64
            else:
                inp0 = int(h_0 * self.f * self.num_blocks)
                inp1 = int(h_1 * self.f * self.num_blocks)

            if layer == 1:
                scaler0 = DeepLabScaler(inp0, int(stride * self.f), activate_f=self.activate_f)
                scaler1 = DeepLabScaler(inp1, int(stride * self.f), activate_f=self.activate_f)
            else:
                scaler0 = DeepLabScaler(inp0, int(stride * self.f), activate_f=self.activate_f)
                scaler1 = DeepLabScaler(inp1, int(stride * self.f), activate_f=self.activate_f)

            h_1 = h_0
            h_0 = h
            self.scalers.append(scaler0)
            self.scalers.append(scaler1)
        self.decoder = build_decoder(cfg, out_strides=stride)
        if cfg.SOLVER.LOSS is not None:
            self.loss_dict = []
            self.loss_weight = []
            for loss_item, loss_weight in zip(cfg.SOLVER.LOSS, cfg.SOLVER.LOSS_WEIGHT):
                if 'ssim' in loss_item or 'grad' in loss_item:
                    self.loss_dict.append(loss_dict[loss_item](channel=cfg.MODEL.IN_CHANNEL))
                else:
                    self.loss_dict.append(loss_dict[loss_item]())
                self.loss_weight.append(loss_weight)

        else:
            self.loss_dict = None
            self.loss_weight = None

    def genotype(self):
        return self.genotpe

    def forward(self, images, targets=None, drop_prob=-1):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")

        h1 = self.stem1(images)
        if self.activate_f.lower() == 'relu':
            h0 = self.stem2(F.relu(h1))
        elif self.activate_f.lower() in ['leaky', 'prelu']:
            h0 = self.stem2(F.leaky_relu(h1, negative_slope=0.2))
        elif self.activate_f.lower() == 'sine':
            h0 = self.stem2(torch.sin(h1))

        if self.endpoint==None:
            endpoint = self.reduce(h0)

        for i, cell in enumerate(self.cells):
            s0 = self.scalers[i*2](h0)
            s1 = self.scalers[i*2+1](h1)
            h1 = h0
            h0 = cell(s0, s1, drop_prob)
            if self.endpoint is not None and i == self.endpoint:
                endpoint = h0

        if self.activate_f.lower() == 'relu':
            pred = self.decoder([endpoint, F.relu(h0)])
        elif self.activate_f.lower() in ['leaky', 'prelu']:
            pred = self.decoder([endpoint, F.leaky_relu(h0, negative_slope=0.2)])
        elif self.activate_f.lower() == 'sine':
            pred= self.decoder([endpoint, torch.sin(h0)])

        if self.use_res:
            pred = images-pred


        if self.training:
            if loss_dict is not None:
                loss = []
                for loss_item, weight in zip(self.loss_dict, self.loss_weight):
                    loss.append(loss_item(pred, targets) * weight)
            else:
                loss = F.mse_loss(pred, targets)
            return pred, {'decoder_loss': sum(loss) / len(loss)}

        else:
            return pred


