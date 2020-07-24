# Proximal
import sys
sys.path.append('./ProxImaL')
from proximal.utils.utils import *
from proximal.utils.metrics import *
from proximal.lin_ops import *
from proximal.prox_fns import *
import cvxpy as cvx
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import random

# bm3d
sys.path.append('./bm3d-3.0.6/examples')
from bm3d import bm3d_rgb, BM3DProfile
from experiment_funcs import get_psnr
from scipy.ndimage.filters import correlate
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm

random.seed(10)

default_cff = 4.0
default_n1 = 8
default_cspace = 0
default_wtransform = 0
default_neiborhood = 8

def generate_dir():
    file_dir = ['./SIDD_crop_bm3d', './SIDD_crop_bm3d/train', './SIDD_crop_bm3d/test', 
        './SIDD_crop_bm3d/train/GT', './SIDD_crop_bm3d/train/NOISY', './SIDD_crop_bm3d/train/RED',
        './SIDD_crop_bm3d/train/PARAM', './SIDD_crop_bm3d/test/GT', './SIDD_crop_bm3d/test/NOISY',
        './SIDD_crop_bm3d/test/RED', './SIDD_crop_bm3d/test/PARAM']

    file_name = glob.glob('./SIDD_crop/*_GT_SRGB')
    order = []
    for elem in file_name:
        order.append(elem.split('/')[-1].split('_')[0])

    for idx in range(len(order)):
        for jdx in range(3, 11):
            file_dir.append(file_dir[jdx]+'/'+order[idx]+'_SRGB')

    for idx in range(len(file_dir)):
        if not os.path.exists(file_dir[idx]):
            os.mkdir(file_dir[idx])

def estimate_the_noise(img):
    I = np.asfortranarray(im2nparray(img))
    I = np.mean(I, axis=2)
    I = np.asfortranarray(I)
    I = np.maximum(I, 0.0)
    ndev = estimate_std(I, 'daub_replicate')
    return ndev

def generate_red_img(noisy_img, pred_psd, gt_img):
    noisy_img = np.array(noisy_img)
    noisy_img = noisy_img / 255.0
    gt_img = np.array(gt_img)
    gt_img = gt_img / 255.0
    profile = BM3DProfile()
    profile.bs_ht = random.choice([2, 4, 8])
    profile.transform_2d_wiener_name = random.choice(['dct', 'dst'])
    profile.bs_wiener = random.choice([4, 5, 6, 7, 8, 9, 10, 11, 12])
    cspace = random.choice(['opp', 'YCbCr'])
    cff = random.uniform(1, 15)

    red_img = bm3d_rgb(noisy_img, cff*pred_psd[0], profile, colorspace=cspace)
    red_img = np.minimum(np.maximum(red_img, 0), 1)

    psnr = get_psnr(gt_img, red_img)

    red_img = Image.fromarray(np.uint8(red_img*255.0))

    return red_img, cff, profile.bs_ht, cspace, profile.transform_2d_wiener_name, profile.bs_wiener, psnr

noisy_dir_list = glob.glob('./SIDD_crop/*_NOISY_SRGB')
gt_dir_list = glob.glob('./SIDD_crop/*_GT_SRGB')
noisy_dir_list.sort()
gt_dir_list.sort()

generate_dir()

for idx in range(1, len(noisy_dir_list)):

    noisy_img_list = glob.glob(noisy_dir_list[idx] + '/*.PNG')
    gt_img_list = glob.glob(gt_dir_list[idx] + '/*.PNG')
    noisy_img_list.sort()
    gt_img_list.sort()

    train_num = int(0.9*len(noisy_img_list))
    train_idx = random.sample(range(len(noisy_img_list)), train_num)

    cnt_train = 0
    cnt_test = 0

    for jdx in range(len(noisy_img_list)):
        noisy_img = Image.open(noisy_img_list[jdx])
        gt_img = Image.open(gt_img_list[jdx])

        # Estimate the noise
        pred_psd = estimate_the_noise(noisy_img)
        
        # five parameter: ['cff', 'n1', 'cspace', 'wtransform', 'neighborhood']
        # ['cff', 'bs_ht', 'YCbCr'or'opp', 'transform_2d_wiener_name', 'bs_wiener']
        red_img, cff, n1, cspace, wtransform, neighborhood, psnr = generate_red_img(noisy_img, pred_psd, gt_img)

        order = noisy_img_list[jdx].split('/')[-2].split('_')[0]

        if jdx in train_idx:
            noisy_img.save('./SIDD_crop_bm3d/train/NOISY/{}_SRGB/{:03d}.PNG'.format(order, cnt_train))
            gt_img.save('./SIDD_crop_bm3d/train/GT/{}_SRGB/{:03d}.PNG'.format(order, cnt_train))
            red_img.save('./SIDD_crop_bm3d/train/RED/{}_SRGB/{:03d}.PNG'.format(order, cnt_train))
            with open('./SIDD_crop_bm3d/train/PARAM/{}_SRGB/{:03d}.txt'.format(order, cnt_train), 'w') as f:
                f.write('{}\n'.format(cff))
                f.write('{}\n'.format(n1))
                f.write('{}\n'.format(cspace))
                f.write('{}\n'.format(wtransform))
                f.write('{}\n'.format(neighborhood))
                f.write('{}\n'.format(psnr))
            f.close()
            cnt_train += 1

        else:
            noisy_img.save('./SIDD_crop_bm3d/test/NOISY/{}_SRGB/{:03d}.PNG'.format(order, cnt_train))
            gt_img.save('./SIDD_crop_bm3d/test/GT/{}_SRGB/{:03d}.PNG'.format(order, cnt_train))
            red_img.save('./SIDD_crop_bm3d/test/RED/{}_SRGB/{:03d}.PNG'.format(order, cnt_train))
            with open('./SIDD_crop_bm3d/test/PARAM/{}_SRGB/{:03d}.txt'.format(order, cnt_train), 'w') as f:
                f.write('{}\n'.format(cff))
                f.write('{}\n'.format(n1))
                f.write('{}\n'.format(cspace))
                f.write('{}\n'.format(wtransform))
                f.write('{}\n'.format(neighborhood))
                f.write('{}\n'.format(psnr))
            f.close()
            cnt_test += 1