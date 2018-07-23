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
    copies = 5
    for data_path in data_paths:
        for i in range(4):
            for img_path in glob.glob(join(data_path, str(i), '*.png')):
                img = cv2.imread(img_path)
                if i in [3]:
                    if random() < 7/(3*copies+7):
                        save_path = join('tmp/data/train', str(i))
                        for k in range(copies):
                            img = augmentation(img)
                            num = 0
                            while os.path.exists(join(save_path, str(num)+'.png')):
                                num += 1
                            cv2.imwrite(join(save_path, str(num)+'.png'),img)
                    else:
                        save_path = join('tmp/data/test', str(i))
                        num = 0
                        while os.path.exists(join(save_path, str(num)+'.png')):
                            num += 1
                        cv2.imwrite(join(save_path, str(num)+'.png'),img)
                else:
                    if random() < 0.7:
                        save_path = join('tmp/data/train', str(i))
                        img = augmentation(img)
                    else:
                        save_path = join('tmp/data/test', str(i))
                    num = 0
                    while os.path.exists(join(save_path, str(num)+'.png')):
                        num += 1
                    cv2.imwrite(join(save_path, str(num)+'.png'),img)

# split_dataset(['/media/weiming/46823A5A823A4F25/data/2018-07-12_140713/annotations-hard',
#             '/media/weiming/46823A5A823A4F25/data/2018-07-10_101334/annotations-hard',
#             '/media/weiming/46823A5A823A4F25/data/2018-06-19_171915/annotations-hard'])

split_dataset(['/media/weiming/46823A5A823A4F25/data/2018-07-19_141101/annotations-hard'])
