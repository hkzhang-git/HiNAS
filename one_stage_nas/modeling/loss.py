import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class log_SSIM_loss(nn.Module):
    def __init__(self, window_size=11, channel=3, is_cuda=True, size_average=True):
        super(log_SSIM_loss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.window = create_window(window_size, channel)
        if is_cuda:
            self.window = self.window.cuda()


    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return -torch.log10(ssim_map.mean())


class negative_SSIM_loss(nn.Module):
    def __init__(self, window_size=11, channel=3, is_cuda=True, size_average=True):
        super(negative_SSIM_loss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.window = create_window(window_size, channel)
        if is_cuda:
            self.window = self.window.cuda()


    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0-ssim_map.mean()


class GRAD_loss(nn.Module):
    def __init__(self, channel=3, is_cuda=True):
        super(GRAD_loss, self).__init__()
        self.edge_conv = nn.Conv2d(channel, channel*2, kernel_size=3, stride=1, padding=1, groups=channel, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = []
        for i in range(channel):
            edge_k.append(edge_kx)
            edge_k.append(edge_ky)

        edge_k = np.stack(edge_k)

        edge_k = torch.from_numpy(edge_k).float().view(channel*2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        for param in self.parameters():
            param.requires_grad = False

        if is_cuda: self.edge_conv.cuda()

    def forward(self, img1, img2):
        img1_grad = self.edge_conv(img1)
        img2_grad = self.edge_conv(img2)

        return F.l1_loss(img1_grad, img2_grad)


class exp_GRAD_loss(nn.Module):
    def __init__(self, channel=3, is_cuda=True):
        super(exp_GRAD_loss, self).__init__()
        self.edge_conv = nn.Conv2d(channel, channel*2, kernel_size=3, stride=1, padding=1, groups=channel, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = []
        for i in range(channel):
            edge_k.append(edge_kx)
            edge_k.append(edge_ky)

        edge_k = np.stack(edge_k)

        edge_k = torch.from_numpy(edge_k).float().view(channel*2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        for param in self.parameters():
            param.requires_grad = False

        if is_cuda: self.edge_conv.cuda()

    def forward(self, img1, img2):
        img1_grad = self.edge_conv(img1)
        img2_grad = self.edge_conv(img2)

        return torch.exp(F.l1_loss(img1_grad, img2_grad)) - 1


class log_PSNR_loss(torch.nn.Module):
    def __init__(self):
        super(log_PSNR_loss, self).__init__()

    def forward(self, img1, img2):
        diff = img1 - img2
        mse = diff*diff.mean()
        return -torch.log10(1.0-mse)


class MSE_loss(torch.nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, img1, img2):
        return F.mse_loss(img1, img2)


class L1_loss(torch.nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()

    def forward(self, img1, img2):
        return F.l1_loss(img1, img2)


loss_dict = {
    'l1': L1_loss,
    'mse': MSE_loss,
    'grad': GRAD_loss,
    'exp_grad': exp_GRAD_loss,
    'log_ssim': log_SSIM_loss,
    'neg_ssim': negative_SSIM_loss,
    'log_psnr': log_PSNR_loss,
}





