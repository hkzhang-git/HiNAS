"""
Implements Auto-DeepLab framework
"""

import torch
from torch import nn
import torch.nn.functional as F

from one_stage_nas.darts.cell import Cell
from one_stage_nas.darts.genotypes import PRIMITIVES
from .decoders import build_decoder
from .common import conv3x3_bn, conv1x1_bn, viterbi
from .loss import loss_dict

class Router_Width(nn.Module):
    """ Propagate hidden states to next layer
    """

    def __init__(self, ind, inp, C, num_strides=4, affine=True):
        """
        Arguments:
            ind (int) [2-5]: index of the cell, which decides output scales
            inp (int): inp size
            C (int): output size of the same scale
        """
        super(Router_Width, self).__init__()
        self.ind = ind
        self.num_strides = num_strides

        if ind > 0:
            # upsample
            self.postprocess0 = conv1x1_bn(inp, C // 2, 1, affine=affine, activate_f=None)

        self.postprocess1 = conv1x1_bn(inp, C, 1, affine=affine, activate_f=None)
        if ind < num_strides - 1:
            # downsample
            self.postprocess2 = conv1x1_bn(inp, C * 2, 1, affine=affine, activate_f=None)

    def forward(self, out):
        """
        Returns:
            h_next ([Tensor]): None for empty
        """
        if self.ind > 0:
            h_next_0 = self.postprocess0(out)
        else:
            h_next_0 = None
        h_next_1 = self.postprocess1(out)
        if self.ind < self.num_strides - 1:
            h_next_2 = self.postprocess2(out)
        else:
            h_next_2 = None
        return h_next_0, h_next_1, h_next_2


class AutoMultiTask(nn.Module):
    """
    Main class for Auto-DeepLab.

    Use one cell per hidden states
    """

    def __init__(self, cfg):
        super(AutoMultiTask, self).__init__()
        self.f = cfg.MODEL.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.num_strides = cfg.MODEL.NUM_STRIDES
        self.primitives = PRIMITIVES[cfg.MODEL.PRIMITIVES]
        self.activatioin_f = cfg.MODEL.ACTIVATION_F
        self.use_res = cfg.MODEL.USE_RES
        affine = cfg.MODEL.AFFINE
        self.stem1 = nn.Sequential(
            conv3x3_bn(cfg.MODEL.IN_CHANNEL, 64, 1, affine=affine, activate_f=self.activatioin_f)
        )
        self.stem2 = conv3x3_bn(64, self.f * self.num_blocks, 1, affine=affine, activate_f=None)
        # generates first h_1
        self.reduce1 = conv3x3_bn(64, self.f, 1, affine=affine, activate_f=None)

        # upsample module for other strides
        self.upsamplers = nn.ModuleList()


        Router = Router_Width
        for i in range(1, self.num_strides):
            self.upsamplers.append(conv1x1_bn(self.f * 2 ** (i - 1),
                                              self.f * 2 ** i,
                                              1, affine=affine, activate_f=None))

        self.cells = nn.ModuleList()
        self.routers = nn.ModuleList()
        self.cell_configs = []
        self.tie_cell = cfg.SEARCH.TIE_CELL

        for l in range(1, self.num_layers + 1):
            for h in range(min(self.num_strides, l + 1)):
                stride = 2 ** h
                C = self.f * stride

                if h < l:
                    self.routers.append(Router(h, C * self.num_blocks,
                                               C, affine=affine))

                self.cell_configs.append(
                    "L{}H{}: {}".format(l, h, C))
                self.cells.append(Cell(self.num_blocks, C,
                                       self.primitives,
                                       affine=affine))

        # ASPP
        self.decoder = build_decoder(cfg)
        self.init_alphas()
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


    def w_parameters(self):
        return [value for key, value in self.named_parameters()
                if 'arch' not in key and value.requires_grad]

    def a_parameters(self):
        a_params = [value for key, value in self.named_parameters() if 'arch' in key]
        return a_params

    def init_alphas(self):
        k = sum(2 + i for i in range(self.num_blocks))
        num_ops = len(self.primitives)
        if self.tie_cell:
            self.arch_alphas = nn.Parameter(torch.ones(k, num_ops))
        else:
            self.arch_alphas = nn.Parameter(torch.ones(self.num_layers, k, num_ops))

        m = sum(min(l+1, self.num_strides) for l in range(self.num_layers))
        beta_weights = torch.ones(m, 3)
        # mask out
        top_inds = []
        btm_inds = []
        start = 0
        for l in range(self.num_layers):
            top_inds.append(start)
            if l+1 < self.num_strides:
                start += l+1
            else:
                start += self.num_strides
                btm_inds.append(start-1)

        beta_weights[top_inds, 0] = -50
        beta_weights[btm_inds, 2] = -50
        self.arch_betas = nn.Parameter(beta_weights)
        self.score_func = F.softmax

    def scores(self):
        return (self.score_func(self.arch_alphas, dim=-1),
                self.score_func(self.arch_betas, dim=-1))

    def forward(self, images, targets=None):

        alphas, betas = self.scores()

        # The first layer is different
        features = self.stem1(images)
        inputs_1 = [self.reduce1(features)]
        if self.activatioin_f.lower() == 'relu':
            features_t = F.relu(features)
        elif self.activatioin_f.lower() in ['leaky', 'prelu']:
            features_t = F.leaky_relu(features, negative_slope=0.2)
        elif self.activatioin_f.lower() == 'sine':
            features_t = torch.sin(features)
        features = self.stem2(features_t)

        hidden_states = [features]

        cell_ind = 0
        router_ind = 0
        for l in range(self.num_layers):
            # prepare next inputs
            inputs_0 = [0] * min(l + 2, self.num_strides)
            for i, hs in enumerate(hidden_states):
                # print('router {}: '.format(router_ind), self.cell_configs[router_ind])
                h_0, h_1, h_2 = self.routers[router_ind](hs)
                # print(h_0 is None, h_1 is None, h_2 is None)
                # print(betas[router_ind])
                if i > 0:
                    inputs_0[i-1] = inputs_0[i-1] + h_0 * betas[router_ind][0]
                inputs_0[i] = inputs_0[i] + h_1 * betas[router_ind][1]
                if i < self.num_strides-1:
                    inputs_0[i+1] = inputs_0[i+1] + h_2 * betas[router_ind][2]
                router_ind += 1

            # run cells
            hidden_states = []
            for i, s0 in enumerate(inputs_0):
                # prepare next input
                if i >= len(inputs_1):
                    # print("using upsampler {}.".format(i-1))
                    inputs_1.append(self.upsamplers[i-1](inputs_1[-1]))
                s1 = inputs_1[i]
                # print('cell: ', self.cell_configs[cell_ind])
                if self.tie_cell:
                    cell_weights = alphas
                else:
                    cell_weights = alphas[l]
                hidden_states.append(self.cells[cell_ind](s0, s1, cell_weights))
                cell_ind += 1

            inputs_1 = inputs_0

        # apply ASPP on hidden_state
        pred = self.decoder(hidden_states)
        if self.use_res:
            pred = images - pred
        pred = torch.sigmoid(pred)

        if self.training:
            if self.loss_dict is not None:
                loss = []
                for loss_item, weight in zip(self.loss_dict, self.loss_weight):
                    loss.append(loss_item(pred, targets) * weight)
            else:
                loss = F.mse_loss(pred, targets)
            return {'decoder_loss': sum(loss) / len(loss)}
        else:
            return pred

    def get_path_genotype(self, betas):
        # construct transition matrix
        trans = []
        b_ind = 0
        for l in range(self.num_layers):
            layer = []
            for i in range(self.num_strides):
                if i < l + 1:
                    layer.append(betas[b_ind].detach().numpy().tolist())
                    b_ind += 1
                else:
                    layer.append([0, 0, 0])
            trans.append(layer)
        return viterbi(trans)

    def genotype(self):
        alphas, betas = self.scores()
        if self.tie_cell:
            gene_cell = self.cells[0].genotype(alphas)
        else:
            gene_cell = []
            for i in range(self.num_layers):
                gene_cell.append(self.cells[0].genotype(alphas[i]))
        gene_path = self.get_path_genotype(betas)
        return gene_cell, gene_path
