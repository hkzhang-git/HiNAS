import json
import numpy as np
import cv2
from PIL import Image

root_dir = '/home/hkzhang/Documents/sdb_a/nas_data/denoise/'
json_dir = './dataset_json/denoise/CBD_real.json'
with open(json_dir, 'r') as f:
    img_dict = json.load(f)

for img_info in img_dict:
    img=Image.open(root_dir + img_info['path_clean'])
    img_name = img_info['path_clean'].split('/')[-1]
    if img.mode != 'RGB':
        print(img_name + ' : ' + img.mode)

