import os
import random
import sys

from skimage.io import imread
import numpy as np

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SRC_PATH, '..', 'data')
YEARBOOK_PATH = os.path.join(DATA_PATH, 'geo')
YEARBOOK_TRAIN_PATH = os.path.join(YEARBOOK_PATH, 'train')
YEARBOOK_TRAIN_LABEL_PATH = os.path.join(YEARBOOK_PATH, 'geo_train.txt')
YEARBOOK_TRAIN_AUG_PATH = os.path.join(YEARBOOK_PATH, 'train_aug')
YEARBOOK_TRAIN_AUG_LABEL_PATH = os.path.join(YEARBOOK_PATH, 'yearbook_train_aug.txt')
YEARBOOK_VALID_PATH = os.path.join(YEARBOOK_PATH, 'valid')
YEARBOOK_VALID_LABEL_PATH = os.path.join(YEARBOOK_PATH, 'geo_valid.txt')
YEARBOOK_MEAN_IMAGE_PATH = os.path.join(YEARBOOK_PATH, 'mean_image.png')
images_final = np.zeros((175010,150,261,3))
def read_labeled_image_list(image_list_file, image_prefixpath, save_prefix):
    save_prefix += '_'
    f = open(image_list_file, 'r')
    images = []
    labels1 = []
    labels2 = []
    c= 0
    n=0
    #gender_labels = []
    for line in f:
           filename, label1,label2 = line.split()
           image_path = os.path.join(image_prefixpath, filename)
           image = imread(image_path)
           images.append(np.pad(image,((55,56),(0,0),(0,0)), 'constant', constant_values=(0)))
           #images = imread(image_path)
           labels1.append(float(label1))
           labels2.append(float(label2))
           c=c+1
           if c == (n+1)*30000 and n<=7 or c==171000:
                np.save(save_prefix+'images_geo'+str(n)+'.npy',images_final)
                np.save(save_prefix+'labels1_geo'+str(n)+'.npy',labels1)
                                                                                                                   1,1           Top
                np.save(save_prefix+'labels2_geo'+str(n)+'.npy',labels2)
                c = 0
                labels1 = []
                labels2 = []
                images = []
                n=n+1

                #break          
       # if 'F' in filename:
       #    gender_labels.append(1)
       # elif 'M' in filename:
       #    gender_labels.append(0)
    print('Found JPEG files across labels',(len(images), len(labels1),len(labels2)))

    # Shuffle
    #shuffled_index = list(range(len(images)))
    #random.seed(12345)
    #random.shuffle(shuffled_index)

    #images = [images[i] for i in shuffled_index]
    #images_final= np.array(images, dtype='float')
    #np.save(save_prefix+'images_geo.npy',images_final)
    #print(images.shape)

    #labels1 = [labels1[i] for i in shuffled_index]
    #labels1x = np.array(labels1)
    #np.save(save_prefix+'labels1_geo.npy',labels1)
    #print(labels1.shape)
    #labels2 = [labels2[i] for i in shuffled_index]
    #labels2x = np.array(labels2)
   # np.save(save_prefix+'labels1_geo.npy',labels2)
    #print(labels2x.shape)
    
    #if len(gender_labels) == len(labels):
    #    gender_labels = [gender_labels[i] for i in shuffled_index]
    #    gender_labels = np.array(gender_labels)
    #    np.save(save_prefix+'gender_labels.npy',gender_labels)
    #    print(gender_labels.shape)


def save_train():
    print('Saving train data as numpy array')
    read_labeled_image_list(YEARBOOK_TRAIN_LABEL_PATH,YEARBOOK_TRAIN_PATH, 'train_geo')

def save_valid():
    print('Saving valid data as numpy array')
    read_labeled_image_list(YEARBOOK_VALID_LABEL_PATH,YEARBOOK_VALID_PATH, 'valid')

save_train()
#save_valid()

                                                                                                                   90,0-1        Bot
