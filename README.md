# HiNAS
Code for paper: 
> [Memory-Efficient Hierarchical Neural Architecture Search for Image Denoising (CVPR 2020)](https://arxiv.org/abs/1909.08228) 

> [Memory-Efficient Hierarchical Neural Architecture Search for Image Restoration](https://arxiv.org/abs/2012.13212)

Compared with the mehtod propsoed in Memory-Efficient Hierarchical Neural Architecture Search for Image Denoising, we made some improvements, which are explained in "log file for DNAS_For_IR"

If you use this code in your paper, please cite our papers
'''
@inproceedings{zhang2020memory,
  title={Memory-efficient hierarchical neural architecture search for image denoising},
  author={Zhang, Haokui and Li, Ying and Chen, Hao and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3657--3666},
  year={2020}
}
'''
## Requirements
'''
This code is tested on Pytorch = 1.0.0.

I build my experimental environment by create a virtual env via anaconda3. 
After activating you env, you can install other all dependences via run: pip install -r requirements.txt. Note that,  to install graphviz, you also need to run: conda install graphviz. 
'''

## Training
