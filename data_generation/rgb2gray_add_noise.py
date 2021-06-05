import os
import numpy as np
from PIL import Image
from glob import glob

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

image_dir = '/home/hkzhang/Documents/sdb_a/nas_data/denoise/BSD500_200'
gray_save_dir = image_dir + '_gray/'

# sigmas=[30, 50, 70]
# for sigma in sigmas:
#     make_if_not_exist(image_dir + '_sigma_{}'.format(sigma))
#
# make_if_not_exist(gray_save_dir)

image_list = glob(os.path.join(image_dir, '*.jpg'))
for im_dir in image_list:
    im_id = im_dir.split('/')[-1][:-4]
    im = Image.open(im_dir).convert('L')
    im.save(gray_save_dir+ im_id + '.png')
    im_clean = np.array(im)

    for sigma in sigmas:
        im_noise = np.array(im_clean + np.random.normal(0, 1, size=im_clean.shape) * sigma, np.uint8)
        im_noise = Image.fromarray(im_noise)
        im_noise.save(image_dir + '_sigma_{}'.format(sigma) + '/{}'.format(im_id) + '.jpg')






