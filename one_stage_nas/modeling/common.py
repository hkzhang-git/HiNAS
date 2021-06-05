import operator

from torch import nn
import torch


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


def conv3x3_bn(inp, oup, stride, affine=True, activate_f='ReLU'):
    if activate_f is None:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup, affine=affine))
    elif activate_f.lower() == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup, affine=affine),
            nn.ReLU(inplace=True)
        )
    elif activate_f.lower() == 'leaky':
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup, affine=affine),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    elif activate_f.lower() == 'prelu':
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup, affine=affine),
            nn.PReLU()
        )
    elif activate_f.lower() == 'sine':
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup, affine=affine),
            Sine()
        )



def conv1x1_bn(inp, oup, stride, affine=True, activate_f='ReLU'):
    if activate_f is None:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
            nn.BatchNorm2d(oup, affine=affine))
    elif activate_f.lower() == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
            nn.BatchNorm2d(oup, affine=affine),
            nn.ReLU(inplace=True)
        )
    elif activate_f.lower() == 'leaky':
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
            nn.BatchNorm2d(oup, affine=affine),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    elif activate_f.lower() == 'prelu':
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
            nn.BatchNorm2d(oup, affine=affine),
            nn.PReLU()
        )
    elif activate_f.lower() == 'sine':
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
            nn.BatchNorm2d(oup, affine=affine),
            Sine()
        )


def sep3x3_bn(inp, oup, rate=1, activate_f='ReLU'):
    if activate_f.lower() == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride=1,
                      padding=rate, dilation=rate, groups=inp,
                      bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True))
    elif activate_f.lower() == 'leaky':
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride=1,
                      padding=rate, dilation=rate, groups=inp,
                      bias=False),
            nn.BatchNorm2d(inp),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(inp, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
    elif activate_f.lower() == 'prelu':
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride=1,
                      padding=rate, dilation=rate, groups=inp,
                      bias=False),
            nn.BatchNorm2d(inp),
            nn.PReLU(),
            nn.Conv2d(inp, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.PReLU())
    elif activate_f.lower() == 'sine':
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride=1,
                      padding=rate, dilation=rate, groups=inp,
                      bias=False),
            nn.BatchNorm2d(inp),
            Sine(),
            nn.Conv2d(inp, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
            Sine())


def viterbi(trans):
    """Dynamic programming to find the most likely path.

    Arguments:
        trans (LxSx3 array)"""
    prob = [1, 0, 0, 0]  # keeps the path with highest prob
    probs = [prob]
    paths = []
    for layer in trans:
        prob_next = [0, 0, 0, 0]
        path = [-1, -1, -1, -1]
        for i, stride in enumerate(layer):
            if i > 0:
                prob_up = stride[0] * prob[i]
                if prob_up > prob_next[i-1]:
                    prob_next[i-1] = prob_up
                    path[i-1] = 0
            prob_same = stride[1] * prob[i]
            if prob_same > prob_next[i]:
                prob_next[i] = prob_same
                path[i] = 1
            if i < 3:
                prob_down = stride[2] * prob[i]
                if prob_down > prob_next[i+1]:
                    prob_next[i+1] = prob_down
                    path[i+1] = 2
        prob = prob_next
        probs.append(prob)
        paths.append(path)

    max_ind, max_prob = max(enumerate(probs[-1]), key=operator.itemgetter(1))

    ml_path = [max_ind]
    for i in range(len(paths) - 1, 0, -1):
        path = paths[i]
        ml_path.insert(0, max_ind - path[max_ind] + 1)
        max_ind = max_ind - path[max_ind] + 1
    print(ml_path)

    # check the prob
    ind = 0
    prob = 1
    for i, layer in enumerate(trans):
        next_ind = ml_path[i]
        stride = layer[ind]
        print(i, layer[ind])
        prob = prob * stride[next_ind-ind+1]
        ind = next_ind

    assert(max_prob - prob < 0.00001)
    return ml_path
