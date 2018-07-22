import numpy as np
import os
from os.path import join
import cv2
import argparse
from random import random, choice, randint
from Utils import read_txt, write_txt, read_csv
from scipy.misc import imresize
from shutil import rmtree, copyfile
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

patch_size = 64
radius = 32

def augmentation(img):
    random_rate = random()
    if random_rate > 0.8:
        R = int(patch_size*random_rate/2)
        shift = int(abs(R-radius+1)/2)
        x_shift = randint(-shift,shift)
        y_shift = randint(-shift,shift)
        img = img[radius-R+y_shift:radius+R+y_shift, radius-R+x_shift:radius+R+x_shift]
        img = img[radius-R:radius+R, radius-R:radius+R]
        img = cv2.resize(img, (patch_size, patch_size))
    if random() > 0.5:
        img = img[:, ::-1]
    return img

def split_dataset(data_paths):
    path = 'tmp/data'
    copies = 5
    if os.path.exists(path):
        rmtree(path)
    for i in range(5):
        os.makedirs(join(path, 'train', str(i)))
        os.makedirs(join(path, 'test', str(i)))
    counts = np.zeros(5)
    # for data_path in data_paths:
    #     for i in range(5):
    #         counts[i] += len(glob.glob(join(data_path, str(i), '*.png')))
    # ratios = min(counts)/counts*2
    for data_path in data_paths:
        for i in range(5):
            for img_path in glob.glob(join(data_path, str(i), '*.png')):
                # if random() < ratios[i]:
                img = cv2.imread(img_path)
                # img2 = np.copy(img)
                # cv2.circle(img2, (32,32), 2, (0,0,255))
                # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                # cv2.imshow('img', img2)
                # key = cv2.waitKey(0)
                if i in [3,4]:
                    if random() < 7/(3*copies+7):
                        save_path = join('tmp/data/train', str(i))
                        for k in range(copies):
                            img2 = augmentation(img)
                            num = len(os.listdir(save_path))
                            cv2.imwrite(join(save_path, str(num)+'.png'),img2)
                    else:
                        save_path = join('tmp/data/test', str(i))
                        num = len(os.listdir(save_path))
                        cv2.imwrite(join(save_path, str(num)+'.png'),img)
                else:
                    if random() < 0.7:
                        save_path = join('tmp/data/train', str(i))
                        img2 = augmentation(img)
                        num = len(os.listdir(save_path))
                        cv2.imwrite(join(save_path, str(num)+'.png'),img2)
                    else:
                        save_path = join('tmp/data/test', str(i))
                        num = len(os.listdir(save_path))
                        cv2.imwrite(join(save_path, str(num)+'.png'),img)

    for img_path in glob.glob(join('/media/weiming/46823A5A823A4F25/data/10/images/*.png')):
        img = cv2.imread(img_path)
        img = img[:,:int(img.shape[1]/2)]
        row, col = img.shape[:2]
        r = choice(range(32,row-32))
        c = choice(range(32,col-32))
        img = img[r-32:r+32,c-32:c+32]
        if random() < 0.7:
            save_path = 'tmp/data/train/0'
            img = augmentation(img)
        else:
            save_path = 'tmp/data/test/0'
        num = len(os.listdir(save_path))
        cv2.imwrite(join(save_path, str(num)+'.png'),img)

# split_dataset(['/media/weiming/46823A5A823A4F25/data/1/annotations-rgb', 
#     '/media/weiming/46823A5A823A4F25/data/2/annotations-rgb', 
#     '/media/weiming/46823A5A823A4F25/data/3/annotations-rgb', 
#     '/media/weiming/46823A5A823A4F25/data/4/annotations-rgb', 
#     '/media/weiming/46823A5A823A4F25/data/4/annotations-rgb2'])

split_dataset(['/media/weiming/46823A5A823A4F25/data/2/annotations',
    '/media/weiming/46823A5A823A4F25/data/2/annotations-1',
    '/media/weiming/46823A5A823A4F25/data/3/annotations', 
    '/media/weiming/46823A5A823A4F25/data/4/annotations', 
    '/media/weiming/46823A5A823A4F25/data/6/annotations',
    '/media/weiming/46823A5A823A4F25/data/8/annotations', 
    '/media/weiming/46823A5A823A4F25/data/10/annotations',
    '/media/weiming/46823A5A823A4F25/data/13/annotations',
    '/media/weiming/46823A5A823A4F25/data/14/annotations'])
