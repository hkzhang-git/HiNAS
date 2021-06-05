"""
DARTS operations
"""
import torch.nn as nn
import torch

# from DCNv2.dcn_v2 import DCN


OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'avg_p_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_p_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect' : lambda C, stride, affine: Identity(),

    #operations with activation function set to ReLU
    'con_c_3x3_relu' : lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 1, dilation=1, affine=affine),
    'con_c_5x5_relu' : lambda C, stride, affine: ReLUConvBN(C, C, 5, stride, 2, dilation=1, affine=affine),
    'sep_c_3x3_relu' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_c_5x5_relu' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_c_3x3_relu' : lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 2, dilation=2, affine=affine),
    'dil_c_5x5_relu' : lambda C, stride, affine: ReLUConvBN(C, C, 5, stride, 4, dilation=2, affine=affine),
    'dil_sc_3x3_2_relu' : lambda C, stride, affine: SepConv(C, C, 3, stride, 2, dilation=2, affine=affine),
    'dil_sc_3x3_4_relu' : lambda C, stride, affine: SepConv(C, C, 3, stride, 4, dilation=4, affine=affine),
    'dil_sc_3x3_8_relu' : lambda C, stride, affine: SepConv(C, C, 3, stride, 8, dilation=8, affine=affine),

    #operations with activation function set to LeakyReLU
    'con_c_3x3_leaky' : lambda C, stride, affine: LeakyConvBN(C, C, 3, stride, 1, dilation=1, affine=affine),
    'con_c_5x5_leaky' : lambda C, stride, affine: LeakyConvBN(C, C, 5, stride, 2, dilation=1, affine=affine),
    'sep_c_3x3_leaky' : lambda C, stride, affine: LeakySepConv(C, C, 3, stride, 1, affine=affine),
    'sep_c_5x5_leaky' : lambda C, stride, affine: LeakySepConv(C, C, 5, stride, 2, affine=affine),
    'dil_c_3x3_leaky' : lambda C, stride, affine: LeakyConvBN(C, C, 3, stride, 2, dilation=2, affine=affine),
    'dil_c_5x5_leaky' : lambda C, stride, affine: LeakyConvBN(C, C, 5, stride, 4, dilation=2, affine=affine),
    'dil_sc_3x3_2_leaky' : lambda C, stride, affine: LeakySepConv(C, C, 3, stride, 2, dilation=2, affine=affine),
    'dil_sc_3x3_4_leaky' : lambda C, stride, affine: LeakySepConv(C, C, 3, stride, 4, dilation=4, affine=affine),
    'dil_sc_3x3_8_leaky' : lambda C, stride, affine: LeakySepConv(C, C, 3, stride, 8, dilation=8, affine=affine),

#operations with activation function set to PReLU
    'con_c_3x3_prelu' : lambda C, stride, affine: PReLUConvBN(C, C, 3, stride, 1, dilation=1, affine=affine),
    'con_c_5x5_prelu' : lambda C, stride, affine: PReLUConvBN(C, C, 5, stride, 2, dilation=1, affine=affine),
    'sep_c_3x3_prelu' : lambda C, stride, affine: PReLUSepConv(C, C, 3, stride, 1, affine=affine),
    'sep_c_5x5_prelu' : lambda C, stride, affine: PReLUSepConv(C, C, 5, stride, 2, affine=affine),
    'dil_c_3x3_prelu' : lambda C, stride, affine: PReLUConvBN(C, C, 3, stride, 2, dilation=2, affine=affine),
    'dil_c_5x5_prelu' : lambda C, stride, affine: PReLUConvBN(C, C, 5, stride, 4, dilation=2, affine=affine),
    'dil_sc_3x3_2_prelu' : lambda C, stride, affine: PReLUSepConv(C, C, 3, stride, 2, dilation=2, affine=affine),
    'dil_sc_3x3_4_prelu' : lambda C, stride, affine: PReLUSepConv(C, C, 3, stride, 4, dilation=4, affine=affine),
    'dil_sc_3x3_8_prelu' : lambda C, stride, affine: PReLUSepConv(C, C, 3, stride, 8, dilation=8, affine=affine),

    #operations with activation function set to Sine
    'con_c_3x3_sine' : lambda C, stride, affine: SineConvBN(C, C, 3, stride, 1, dilation=1, affine=affine),
    'con_c_5x5_sine' : lambda C, stride, affine: SineConvBN(C, C, 5, stride, 2, dilation=1, affine=affine),
    'sep_c_3x3_sine' : lambda C, stride, affine: SineSepConv(C, C, 3, stride, 1, affine=affine),
    'sep_c_5x5_sine' : lambda C, stride, affine: SineSepConv(C, C, 5, stride, 2, affine=affine),
    'dil_c_3x3_sine' : lambda C, stride, affine: SineConvBN(C, C, 3, stride, 2, dilation=2, affine=affine),
    'dil_c_5x5_sine' : lambda C, stride, affine: SineConvBN(C, C, 5, stride, 4, dilation=2, affine=affine),
    'dil_sc_3x3_2_sine' : lambda C, stride, affine: SineSepConv(C, C, 3, stride, 2, dilation=2, affine=affine),
    'dil_sc_3x3_4_sine' : lambda C, stride, affine: SineSepConv(C, C, 3, stride, 4, dilation=4, affine=affine),
    'dil_sc_3x3_8_sine' : lambda C, stride, affine: SineSepConv(C, C, 3, stride, 8, dilation=8, affine=affine),
}


class ReLUConvBN(nn.Module):
    """not used"""

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 dilation=1, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class LeakyConvBN(nn.Module):
    """not used"""

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 dilation=1, affine=True):
        super(LeakyConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class PReLUConvBN(nn.Module):
    """not used"""

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 dilation=1, affine=True):
        super(PReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SineConvBN(nn.Module):
    """not used"""

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 dilation=1, affine=True):
        super(SineConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(torch.sin(x))


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1,
                 affine=True, repeats=2):
        super(SepConv, self).__init__()
        basic_op = lambda: nn.Sequential(
          nn.ReLU(inplace=False),
          nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=C_in,
                    bias=False),
          nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm2d(C_in, affine=affine),
        )
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                               basic_op())

    def forward(self, x):
        return self.op(x)


class LeakySepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1,
                 affine=True, repeats=2):
        super(LeakySepConv, self).__init__()
        basic_op = lambda: nn.Sequential(
          nn.LeakyReLU(negative_slope=0.2, inplace=False),
          nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=C_in,
                    bias=False),
          nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm2d(C_in, affine=affine),
        )
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                               basic_op())

    def forward(self, x):
        return self.op(x)


class PReLUSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1,
                 affine=True, repeats=2):
        super(PReLUSepConv, self).__init__()
        basic_op = lambda: nn.Sequential(
          nn.PReLU(),
          nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=C_in,
                    bias=False),
          nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm2d(C_in, affine=affine),
        )
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                               basic_op())

    def forward(self, x):
        return self.op(x)


class SineSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1,
                 affine=True, repeats=2):
        super(SineSepConv, self).__init__()
        basic_op = lambda: nn.Sequential(
          nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=C_in,
                    bias=False),
          nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm2d(C_in, affine=affine),
        )
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                               basic_op())

    def forward(self, x):
        return self.op(torch.sin(x))


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)


# class DefConv(nn.Module):
#
#     def __init__(self, C_in, C_out, ksize, affine=True):
#         super(DefConv, self).__init__()
#         self.dcn = nn.Sequential(nn.ReLU(inplace=False),
#                                  DCN(C_in, C_out, ksize, stride=1,
#                                      padding=ksize // 2, deformable_groups=2),
#                                  nn.BatchNorm2d(C_out, affine=affine))
#
#     def forward(self, x):
#         return self.dcn(x)
