import torch
import numpy as np
import torch.nn.functional as F
from math import exp

# PSNR
class PSNR(object):
    def __init__(self):
        self.sum_psnr = 0
        self.im_count = 0

    def __call__(self, output, gt):

        output = output*255.0
        gt = gt*255.0
        diff = (output - gt)
        mse = torch.mean(diff*diff)
        psnr = float(10*torch.log10(255.0*255.0/mse))

        self.sum_psnr = self.sum_psnr + psnr
        self.im_count += 1.0

    def metric_get(self, frac=4):
        return round(self.sum_psnr/self.im_count, frac)

    def reset(self):
        self.sum_psnr = 0
        self.im_count = 0


def gaussian(window_size=11, sigma=1.5):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size=11, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


#SSIM
class SSIM(object):
    def __init__(self, window_size=11, channel=3, is_cuda=True):
        if is_cuda:
            self.window = create_window(window_size, channel).to('cuda')
        else:
            self.window = create_window(window_size, channel).to('cpu')

        self.window_size = window_size
        self.channel = channel
        self.sum_ssim = 0
        self.im_count = 0

    def __call__(self, output, gt, transpose=True):
        if transpose:
            output = output.transpose(0, 1).transpose(0, 2).unsqueeze(0)
            gt = gt.transpose(0, 1).transpose(0, 2).unsqueeze(0)

        mu1 = F.conv2d(output, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(gt, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(output * output, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(gt * gt, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(output * gt, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        self.sum_ssim = self.sum_ssim + float(ssim_map.mean())
        self.im_count += 1.0


    def metric_get(self, frac=4):
        return round(self.sum_ssim/self.im_count, frac)

    def reset(self):
        self.sum_ssim = 0
        self.im_count = 0


metric_dict = {
    'PSNR': PSNR,
    'SSIM': SSIM
}

