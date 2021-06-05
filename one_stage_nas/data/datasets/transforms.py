from PIL import Image
import random
import torch
import math
import numpy as np
import matplotlib.pyplot as plt


# # implemented with scipy.ndimage
# class RandomRotate(object):
#     """Random rotation of the image from -angle to angle (in degrees).
#     """
#
#     def __init__(self, angle=10, order=2, reshape=False):
#         self.angle = angle
#         self.reshape = reshape
#         self.order = order
#
#     def __call__(self, sample):
#         image, target = sample['image'], sample['target']
#
#         applied_angle = random.uniform(-self.angle, self.angle)
#         angle1 = applied_angle
#
#         image = ndimage.interpolation.rotate(
#             image, angle1, reshape=self.reshape, order=self.order)
#         target = ndimage.interpolation.rotate(
#             target, angle1, reshape=self.reshape, order=self.order)
#
#         image = Image.fromarray(image)
#         target = Image.fromarray(target)
#
#         return {'image': image, 'target': target}


# implemented with PIL.Image
class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees).
    """

    def __init__(self, angle=10):
        self.angle = angle

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        applied_angle = random.randint(-self.angle, self.angle)

        return {'image': image.rotate(applied_angle), 'target': target.rotate(applied_angle)}


class FourLRotate(object):
    """
    Four level random rotation
    Random rotations [0, 90, 180, 270]
    """

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        applied_angle = random.randint(0, 3)

        if applied_angle == 0:
            return sample
        elif applied_angle == 1:
            return {'image': image.rotate(90), 'target': target.rotate(90)}
        elif applied_angle == 2:
            return {'image': image.rotate(180), 'target': target.rotate(180)}
        elif applied_angle == 3:
            return {'image': image.rotate(270), 'target': target.rotate(270)}


class RandomCrop(object):
    """Rescale the input PIL.Image to the given size.
       Default interpolation method  is ``PIL.Image.BILINEAR``
    """

    def __init__(self, crop_size, center=False):
        self.crop_size = crop_size
        self.center = center

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        h = image.height
        w = image.width
        if self.center is not True:
            crop_loc_h = random.uniform(0, 1)
            crop_loc_w = random.uniform(0, 1)
        else:
            crop_loc_h = 0.5
            crop_loc_w = 0.5

        h_start = math.floor((h-self.crop_size)*crop_loc_h)
        w_start = math.floor((w-self.crop_size)*crop_loc_w)
        h_end = h_start + self.crop_size
        w_end = w_start + self.crop_size
        crop_position = (w_start, h_start, w_end, h_end)

        image = image.crop(crop_position)
        target = target.crop(crop_position)

        return {'image': image, 'target': target}


class Sr_RandomCrop(object):
    """Rescale the input PIL.Image to the given size.
       Default interpolation method  is ``PIL.Image.BILINEAR``
    """

    def __init__(self, crop_size, s_factor):
        self.crop_size = crop_size
        self.s_factor = s_factor

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        h = image.height
        w = image.width
        crop_loc_h = random.uniform(0, 1)
        crop_loc_w = random.uniform(0, 1)
        h_start = math.floor((h-self.crop_size)*crop_loc_h)
        w_start = math.floor((w-self.crop_size)*crop_loc_w)
        h_end = h_start + self.crop_size
        w_end = w_start + self.crop_size
        image_crop_position = (w_start, h_start, w_end, h_end)
        target_crop_position = (w_start*self.s_factor, h_start*self.s_factor, w_end*self.s_factor, h_end*self.s_factor)

        image = image.crop(image_crop_position)
        target = target.crop(target_crop_position)

        return {'image': image, 'target': target}


class RandomRescaleCrop(object):
    """Rescale the input PIL.Image to the given size.
       Default interpolation method  is ``PIL.Image.BILINEAR``
    """

    def __init__(self, crop_size, scale=[0.8, 1.2], prob=0.5):
        self.crop_size = crop_size
        self.scale = scale
        self.prob = prob

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        if random.random() > self.prob:
            scale_factor = 1.0
        else:
            scale_factor = random.uniform(self.scale[0], self.scale[1])

        h = math.floor(image.height * scale_factor)
        w = math.floor(image.width * scale_factor)

        crop_loc_h = random.uniform(0, 1)
        crop_loc_w = random.uniform(0, 1)

        h_start = math.floor((h-self.crop_size)*crop_loc_h)
        w_start = math.floor((w-self.crop_size)*crop_loc_w)
        h_end = h_start + self.crop_size
        w_end = w_start + self.crop_size
        crop_position = (w_start, h_start, w_end, h_end)

        image = image.resize((w, h), Image.BICUBIC)
        target = target.resize((w, h), Image.BICUBIC)

        image = image.crop(crop_position)
        target = target.crop(crop_position)

        return {'image': image, 'target': target}


class RandomMirror(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'target': target}


class RandomOverturn(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            target = target.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': image, 'target': target}


class Normalize(object):
    """Normalise an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, scale=255.0, mean=0.5, std=0.5):
        self.scale = scale
        # self.mean = torch.tensor(mean).view((3, 1, 1))
        # self.std = torch.tensor(std).view((3, 1, 1))
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']/self.scale
        target = sample['target']/self.scale
        return {'image': (image - self.mean) / self.std, 'target': target}


class Rescale(object):
    """Normalise an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, scale=255.0):
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']/self.scale
        target = sample['target']/self.scale
        return {'image': image, 'target': target}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        image = self.PIL2Tensor(image)
        target = self.PIL2Tensor(target)

        return {'image': image.float(), 'target': target.float()}

    def PIL2Tensor(self, pic):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode in ['L']:
            channel = 1
        elif pic.mode in ['RGB']:
            channel = 3
        img = img.view(pic.size[1], pic.size[0], channel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        return img


class NoiseToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, sigma=[]):
        self.sigma = sigma
        self.sigma_len = len(sigma)

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        if self.sigma_len != 0:
            random_id = random.randint(0, self.sigma_len-1)
            image = self.PIL2Tensor(image)
            noise = torch.randn(image.size()).mul_(self.sigma[random_id])
            image = image.float() + noise
        else:
            image = self.PIL2Tensor(image)
        target = self.PIL2Tensor(target)

        return {'image': image.float(), 'target': target.float()}

    def PIL2Tensor(self, pic):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode in ['L']:
            channel = 1
        elif pic.mode in ['RGB']:
            channel = 3
        img = img.view(pic.size[1], pic.size[0], channel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        return img


class Ndarray2tensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        arw, rgb, gt = sample['arw'], sample['rgb'], sample['gt']

        arw = arw.transpose((2, 0, 1))
        rgb = rgb.transpose((2, 0, 1))
        gt = gt.transpose((2, 0, 1))

        arw, rgb, gt = np.ascontiguousarray(arw), np.ascontiguousarray(rgb), np.ascontiguousarray(gt)
        arw, rgb, gt = torch.from_numpy(arw), torch.from_numpy(rgb), torch.from_numpy(gt)

        return {'arw': arw, 'rgb': rgb, 'gt': gt}


class SID_RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.crop_size = output_size

    def __call__(self, sample):
        arw, rgb, gt = sample['arw'], sample['rgb'], sample['gt']
        h, w = arw.shape[0], arw.shape[1]
        np.random.seed()
        xx = np.random.randint(0, w - self.crop_size)
        yy = np.random.randint(0, h - self.crop_size)
        arw = arw[yy:yy + self.crop_size, xx:xx + self.crop_size, :]
        rgb = rgb[yy * 2:yy * 2 + self.crop_size * 2, xx * 2:xx * 2 + self.crop_size * 2, :]
        gt = gt[yy * 2:yy * 2 + self.crop_size * 2, xx * 2:xx * 2 + self.crop_size * 2, :]

        return {'arw': arw, 'rgb': rgb,  'gt': gt}


class SID_RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        arw, rgb, gt = sample['arw'], sample['rgb'], sample['gt']

        do_reflection = np.random.randint(2)
        do_mirror = np.random.randint(2)
        do_transpose = np.random.randint(2)
        if do_reflection:
            arw = np.flip(arw, 0)
            rgb = np.flip(rgb, 0)
            gt = np.flip(gt, 0)
        if do_mirror:
            arw = np.flip(arw, 1)
            rgb = np.flip(rgb, 1)
            gt = np.flip(gt, 1)
        if do_transpose:
            arw = np.transpose(arw, (1, 0, 2))
            rgb = np.transpose(rgb, (1, 0, 2))
            gt = np.transpose(gt, (1, 0, 2))

        return {'arw': arw, 'rgb': rgb, 'gt': gt}


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


