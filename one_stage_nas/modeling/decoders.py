import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .common import conv1x1_bn, conv3x3_bn, sep3x3_bn


class ASPPModule(nn.Module):
    """ASPP module of DeepLab V3+. Using separable atrous conv.
    Currently no GAP. Don't think GAP is useful for cityscapes.
    """

    def __init__(self, inp, oup, rates, affine=True, use_gap=True, activate_f='ReLU'):
        super(ASPPModule, self).__init__()
        self.conv1 = conv1x1_bn(inp, oup, 1, affine=affine, activate_f=activate_f)
        self.atrous = nn.ModuleList()
        self.use_gap = use_gap
        for rate in rates:
            self.atrous.append(sep3x3_bn(inp, oup, rate, activate_f=activate_f))
        num_branches = 1 + len(rates)
        if use_gap:
            self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     conv1x1_bn(inp, oup, 1, activate_f=activate_f))
            num_branches += 1
        self.conv_last = conv1x1_bn(oup * num_branches,
                                    oup, 1, affine=affine, activate_f=activate_f)

    def forward(self, x):
        atrous_outs = [atrous(x) for atrous in self.atrous]
        atrous_outs.append(self.conv1(x))
        if self.use_gap:
            gap = self.gap(x)
            gap = F.interpolate(gap, size=x.size()[2:],
                                mode='bilinear', align_corners=False)
            atrous_outs.append(gap)
        x = torch.cat(atrous_outs, dim=1)
        x = self.conv_last(x)
        return x


# Decoders for learning to see in the dark
class Sr_Decoder(nn.Module):
    """DeepLab V3+ decoder
    """

    def __init__(self, cfg, out_stride):
        super(Sr_Decoder, self).__init__()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.use_aspp = cfg.MODEL.USE_ASPP
        self.s_factor = cfg.DATALOADER.S_FACTOR
        BxF = cfg.MODEL.NUM_BLOCKS * cfg.MODEL.FILTER_MULTIPLIER
        inp = int(BxF * out_stride)
        rates = cfg.MODEL.ASPP_RATES
        if self.use_aspp:
            self.pre_conv = ASPPModule(inp, 128, rates, use_gap=False, activate_f=self.activate_f)
        else:
            self.pre_conv = conv1x1_bn(inp, 128, 1, activate_f=self.activate_f)
        self.proj = conv1x1_bn(BxF, 48, 1, activate_f=self.activate_f)
        self.conv0 = sep3x3_bn(176, 128, activate_f=self.activate_f)
        self.conv1 = sep3x3_bn(128, 128, activate_f=self.activate_f)
        self.clf = nn.Sequential(
            nn.Conv2d(128, 3 * self.s_factor*self.s_factor, 3, padding=1),
            nn.PixelShuffle(upscale_factor=self.s_factor)
        )

    def forward(self, x, targets=None, loss_dict=None, loss_weight=None):
        x0, x1 = x
        x1 = self.pre_conv(x1)
        x = torch.cat((self.proj(x0), x1), dim=1)
        x = self.conv1(self.conv0(x))
        pred = self.clf(x)
        return pred


class Sr_AutoDecoder(nn.Module):
    """ ASPP Module at each output features
    """

    def __init__(self, cfg, out_strides):
        super(Sr_AutoDecoder, self).__init__()
        self.aspps = nn.ModuleList()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.s_factor = cfg.DATALOADER.S_FACTOR
        self.ws_factors = cfg.MODEL.WS_FACTORS
        BxF = cfg.MODEL.NUM_BLOCKS * cfg.MODEL.FILTER_MULTIPLIER
        affine = cfg.MODEL.AFFINE
        num_strides = len(out_strides)
        for i, out_stride in enumerate(out_strides):
            rate = out_stride
            inp = int(BxF * self.ws_factors[i])

            oup = BxF
            self.aspps.append(ASPPModule(inp, oup, [rate],
                                         affine=affine,
                                         use_gap=False, activate_f = self.activate_f))
        self.pre_cls = conv3x3_bn(BxF * num_strides,
                                  BxF * num_strides,
                                  1, affine=affine, activate_f = self.activate_f )
        self.clf = nn.Sequential(
            nn.Conv2d(BxF * num_strides, 3*self.s_factor*self.s_factor, 3, padding=1),
            nn.PixelShuffle(upscale_factor=self.s_factor)
        )


    def forward(self, x):
        l1_size = x[0].size()[2:]
        x = [aspp(x_i) for aspp, x_i in zip(self.aspps, x)]
        x = [F.interpolate(x_i, size=l1_size, mode='bilinear') if i > 0 else x_i
             for i, x_i in enumerate(x)]
        x = self.pre_cls(torch.cat(x, dim=1))
        pred = self.clf(x)
        return pred


# Decoders for learning to see in the dark
class Sid_Decoder(nn.Module):
    """DeepLab V3+ decoder
    """

    def __init__(self, cfg, out_stride):
        super(Sid_Decoder, self).__init__()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.use_aspp = cfg.MODEL.USE_ASPP
        BxF = cfg.MODEL.NUM_BLOCKS * cfg.MODEL.FILTER_MULTIPLIER
        inp = BxF * out_stride
        rates = cfg.MODEL.ASPP_RATES
        if self.use_aspp:
            self.pre_conv = ASPPModule(inp, 128, rates, use_gap=False, activate_f=self.activate_f)
        else:
            self.pre_conv = conv1x1_bn(inp, 128, 1, activate_f=self.activate_f)
        self.proj = conv1x1_bn(BxF, 48, 1, activate_f=self.activate_f)
        self.conv0 = sep3x3_bn(176, 128, activate_f=self.activate_f)
        self.conv1 = sep3x3_bn(128, 128, activate_f=self.activate_f)
        self.clf = nn.Sequential(
            nn.Conv2d(128, 12, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x, targets=None, loss_dict=None, loss_weight=None):
        x0, x1 = x
        x1 = self.pre_conv(x1)
        x = torch.cat((self.proj(x0), x1), dim=1)
        x = self.conv1(self.conv0(x))
        pred = self.clf(x)
        return pred


class Sid_AutoDecoder(nn.Module):
    """ ASPP Module at each output features
    """

    def __init__(self, cfg, out_strides):
        super(Sid_AutoDecoder, self).__init__()
        self.aspps = nn.ModuleList()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        BxF = cfg.MODEL.NUM_BLOCKS * cfg.MODEL.FILTER_MULTIPLIER
        affine = cfg.MODEL.AFFINE
        num_strides = len(out_strides)
        for i, out_stride in enumerate(out_strides):
            rate = out_stride
            inp = int(BxF * np.power(2, i))

            oup = BxF
            self.aspps.append(ASPPModule(inp, oup, [rate],
                                         affine=affine,
                                         use_gap=False, activate_f = self.activate_f))
        self.pre_cls = conv3x3_bn(BxF * num_strides,
                                  BxF * num_strides,
                                  1, affine=affine, activate_f = self.activate_f )
        self.clf = nn.Sequential(
            nn.Conv2d(BxF * num_strides, 12, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x):
        l1_size = x[0].size()[2:]
        x = [aspp(x_i) for aspp, x_i in zip(self.aspps, x)]
        x = [F.interpolate(x_i, size=l1_size, mode='bilinear') if i > 0 else x_i
             for i, x_i in enumerate(x)]
        x = self.pre_cls(torch.cat(x, dim=1))
        pred = self.clf(x)
        return pred


# Decoders for single image denoising
class Dn_Decoder(nn.Module):
    """DeepLab V3+ decoder
    """

    def __init__(self, cfg, out_stride):
        super(Dn_Decoder, self).__init__()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.use_aspp = cfg.MODEL.USE_ASPP
        BxF = cfg.MODEL.NUM_BLOCKS * cfg.MODEL.FILTER_MULTIPLIER
        inp = int(BxF * out_stride)
        rates = cfg.MODEL.ASPP_RATES
        if self.use_aspp:
            self.pre_conv = ASPPModule(inp, 128, rates, use_gap=False, activate_f=self.activate_f)
        else:
            self.pre_conv = conv1x1_bn(inp, 128, 1, activate_f=self.activate_f)
        self.proj = conv1x1_bn(BxF, 48, 1, activate_f=self.activate_f)
        self.conv0 = sep3x3_bn(176, 128, activate_f=self.activate_f)
        self.conv1 = sep3x3_bn(128, 128, activate_f=self.activate_f)
        self.clf = nn.Conv2d(128, cfg.MODEL.IN_CHANNEL, 3, padding=1)

    def forward(self, x, targets=None, loss_dict=None, loss_weight=None):
        x0, x1 = x
        x1 = self.pre_conv(x1)
        x = torch.cat((self.proj(x0), x1), dim=1)
        x = self.conv1(self.conv0(x))
        pred = self.clf(x)
        return pred


class Dn_AutoDecoder(nn.Module):
    """ ASPP Module at each output features
    """

    def __init__(self, cfg, out_strides):
        super(Dn_AutoDecoder, self).__init__()
        self.aspps = nn.ModuleList()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.ws_factors = cfg.MODEL.WS_FACTORS
        BxF = cfg.MODEL.NUM_BLOCKS * cfg.MODEL.FILTER_MULTIPLIER
        affine = cfg.MODEL.AFFINE
        num_strides = len(out_strides)
        for i, out_stride in enumerate(out_strides):
            rate = out_stride
            inp = int(BxF * self.ws_factors[i])

            oup = BxF
            self.aspps.append(ASPPModule(inp, oup, [rate],
                                         affine=affine,
                                         use_gap=False, activate_f = self.activate_f))
        self.pre_cls = conv3x3_bn(BxF * num_strides,
                                  BxF * num_strides,
                                  1, affine=affine, activate_f = self.activate_f )
        self.clf = nn.Conv2d(BxF * num_strides, cfg.MODEL.IN_CHANNEL, 3, padding=1)

    def forward(self, x):
        l1_size = x[0].size()[2:]
        x = [aspp(x_i) for aspp, x_i in zip(self.aspps, x)]
        x = [F.interpolate(x_i, size=l1_size, mode='bilinear') if i > 0 else x_i
             for i, x_i in enumerate(x)]
        x = self.pre_cls(torch.cat(x, dim=1))
        pred = self.clf(x)
        return pred


Auto_Decoders = {
    "sr": Sr_AutoDecoder,
    "sid": Sid_AutoDecoder,
    "dn": Dn_AutoDecoder,
}


Decoders = {
    "sr": Sr_Decoder,
    "sid": Sid_Decoder,
    "dn": Dn_Decoder,
}


def build_decoder(cfg, out_strides=[4, 8, 16, 32]):
    """
    out_stride (int or List)
    """
    if cfg.SEARCH.SEARCH_ON:
        out_strides = np.ones(cfg.MODEL.NUM_STRIDES, np.int16) * 16
        return Auto_Decoders[cfg.DATASET.TASK](cfg, out_strides)
    else:
        return Decoders[cfg.DATASET.TASK](cfg, out_strides)
