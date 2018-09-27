#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
from os import path
#import os
import random
import sys
#import threading

from skimage.io import imread
import numpy as np
#import tensorflow as tf

SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')
YEARBOOK_PATH = path.join(DATA_PATH, 'yearbook')
YEARBOOK_TRAIN_PATH = path.join(YEARBOOK_PATH, 'train')
YEARBOOK_TRAIN_LABEL_PATH = path.join(YEARBOOK_PATH, 'yearbook_train.txt')
YEARBOOK_TRAIN_AUG_PATH = path.join(YEARBOOK_PATH, 'train_aug')
YEARBOOK_VALID_PATH = path.join(YEARBOOK_PATH, 'valid')
YEARBOOK_TRAIN_AUG_LABEL_PATH = path.join(YEARBOOK_PATH, 'yearbook_train_aug.txt')
YEARBOOK_VALID_LABEL_PATH = path.join(YEARBOOK_PATH, 'yearbook_valid.txt')

def read_labeled_image_list(image_list_file,image_prefixpath):
    f = open(image_list_file, 'r')
    images = []
    labels = []
    #c = 0
    for line in f:
        filename, label = line.split()
        images.append(imread(image_prefixpath+'/'+filename))
        #if int(label) not in labels:
        labels.append(int(label))
        #c += 1        
        #if c > 10:
        #    break
    shuffled_index = list(range(len(images)))
    shuffled_index_labels = list(range(len(labels)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    images = [images[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found JPEG files across labels',(len(images), len(labels)))
    images = np.array(images)
    labels = np.array(labels)
    np.save('images.npy',images)
    np.save('labels.npy',labels)
    #labels = labels - np.min(labels)
    print(labels.shape)
    print(images.shape)
    return images, labels

def load_train():
  # Run it!
    print('load_train called')
    return read_labeled_image_list(YEARBOOK_TRAIN_AUG_LABEL_PATH,YEARBOOK_TRAIN_AUG_PATH)

load_train()
