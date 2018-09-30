import os
import random
import sys

from skimage.io import imread
import numpy as np

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SRC_PATH, '..', 'data')
YEARBOOK_PATH = os.path.join(DATA_PATH, 'yearbook')
YEARBOOK_TRAIN_PATH = os.path.join(YEARBOOK_PATH, 'train')
YEARBOOK_TRAIN_LABEL_PATH = os.path.join(YEARBOOK_PATH, 'yearbook_train.txt')
YEARBOOK_TRAIN_AUG_PATH = os.path.join(YEARBOOK_PATH, 'train_aug')
YEARBOOK_TRAIN_AUG_LABEL_PATH = os.path.join(YEARBOOK_PATH, 'yearbook_train_aug.txt')
YEARBOOK_VALID_PATH = os.path.join(YEARBOOK_PATH, 'valid')
YEARBOOK_VALID_LABEL_PATH = os.path.join(YEARBOOK_PATH, 'yearbook_valid.txt')
YEARBOOK_MEAN_IMAGE_PATH = os.path.join(YEARBOOK_PATH, 'mean_image.png')

def read_labeled_image_list(image_list_file, image_prefixpath, save_prefix):
    save_prefix += '_'
    f = open(image_list_file, 'r')
    images = []
    labels = []
    gender_labels = []
    for line in f:
        filename, label = line.split()
        image_path = os.path.join(image_prefixpath, filename)
        image = imread(image_path)
        images.append(np.pad(image,((7,8),(0,0),(0,0)), 'constant', constant_values=(0)))
        labels.append(int(label))
        if 'F' in filename:
            gender_labels.append(1)
        elif 'M' in filename:
            gender_labels.append(0)
    print('Found JPEG files across labels',(len(images), len(labels)))
    
    # Shuffle
    shuffled_index = list(range(len(images)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    images = [images[i] for i in shuffled_index]
    images = np.array(images, dtype='float')
    np.save(save_prefix+'images.npy',images)
    print(images.shape)
    
    labels = [labels[i] for i in shuffled_index]
    labels = np.array(labels)
    np.save(save_prefix+'labels.npy',labels)
    print(labels.shape)
    
    if len(gender_labels) == len(labels):
        gender_labels = [gender_labels[i] for i in shuffled_index]
        gender_labels = np.array(gender_labels)
        np.save(save_prefix+'gender_labels.npy',gender_labels)
        print(gender_labels.shape)
        

def save_train():
    print('Saving train data as numpy array')
    read_labeled_image_list(YEARBOOK_TRAIN_LABEL_PATH,YEARBOOK_TRAIN_PATH, 'train_non-aug')

def save_train_aug():
    print('Saving train_aug data as numpy array')
    read_labeled_image_list(YEARBOOK_TRAIN_AUG_LABEL_PATH,YEARBOOK_TRAIN_AUG_PATH, 'train_aug')

def save_valid():
    print('Saving valid data as numpy array')
    read_labeled_image_list(YEARBOOK_VALID_LABEL_PATH,YEARBOOK_VALID_PATH, 'valid')

save_train()
save_train_aug()
save_valid()

