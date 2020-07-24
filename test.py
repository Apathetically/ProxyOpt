import glob
import numpy as np
from numpy import *

def get_ave_psnr_net():
    all_psnr_list = []
    srgb_dir = glob.glob('./SIDD_crop_net/test/PARAM/*_SRGB')
    for dir in srgb_dir:
        txts = glob.glob('{}/*.txt'.format(dir))
        for txt in txts:
            f = open(txt, 'r')
            line = f.readline()
            for idx in range(5):
                line = f.readline()
            all_psnr_list.append(float(line))

    srgb_dir = glob.glob('./SIDD_crop_net/train/PARAM/*_SRGB')
    for dir in srgb_dir:
        txts = glob.glob('{}/*.txt'.format(dir))
        for txt in txts:
            f = open(txt, 'r')
            line = f.readline()
            for idx in range(5):
                line = f.readline()
            all_psnr_list.append(float(line))

    return mean(all_psnr_list)

def get_ave_psnr_rand():
    all_psnr_list = []
    srgb_dir = glob.glob('./SIDD_crop_bm3d/test/PARAM/*_SRGB')
    for dir in srgb_dir:
        txts = glob.glob('{}/*.txt'.format(dir))
        for txt in txts:
            f = open(txt, 'r')
            line = f.readline()
            for idx in range(5):
                line = f.readline()
            all_psnr_list.append(float(line))

    srgb_dir = glob.glob('./SIDD_crop_bm3d/train/PARAM/*_SRGB')
    for dir in srgb_dir:
        txts = glob.glob('{}/*.txt'.format(dir))
        for txt in txts:
            f = open(txt, 'r')
            line = f.readline()
            for idx in range(5):
                line = f.readline()
            all_psnr_list.append(float(line))

    return mean(all_psnr_list)

if __name__ == '__main__':
    print('random parameter: {:.5f}  optimized parameyer: {:.5f}'.format(get_ave_psnr_rand(), get_ave_psnr_net()))