from glob import glob
from PIL import Image
import os


set = 'sigma70'
name_source = '/home/hkzhang/Documents/sdb_a/nas_data/denoise/BSD500_200'
image_source = '/home/hkzhang/Documents/codes/Architecture_search/projects/comparison_methods/denoise/n3net-master/src_denoising/BSD500_result'
save_dir = image_source + '/{}_resort'.format(set)

name_source_list = glob(name_source + '/*.jpg')
name_list = [item.split('/')[-1].split('.')[0] for item in name_source_list]
name_list.sort()

image_list = glob(os.path.join(image_source, set, '*.jpg'))
image_list.sort()
for im_dir, new_id in zip(image_list, name_list):
    im=Image.open(im_dir)
    im.save(save_dir + '/{}.jpg'.format(new_id))
