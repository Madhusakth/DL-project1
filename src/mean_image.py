from __future__ import print_function
from os import path, listdir, unlink
import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from util import *


SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')
YEARBOOK_PATH = path.join(DATA_PATH, 'yearbook')
YEARBOOK_TRAIN_PATH = path.join(YEARBOOK_PATH, 'train')
YEARBOOK_TRAIN_LABEL_PATH = path.join(YEARBOOK_PATH, 'yearbook_train.txt')

YEARBOOK_TRAIN_AUG_PATH = path.join(YEARBOOK_PATH, 'train_aug')
YEARBOOK_TRAIN_AUG_LABEL_PATH = path.join(YEARBOOK_PATH, 'yearbook_train_aug.txt')


def mean_image():
	train_list = listYearbook(train=True, valid=False)
	mean_img = None
	image_array = []
	for image_gr_truth in train_list:
		image_path = path.join(YEARBOOK_TRAIN_PATH, image_gr_truth[0])
		img=imread(image_path)
		image_array.append(img)
	image_array = np.array(image_array)
	mean_img = np.mean(image_array, axis=0)
	mean_image_path = path.join(YEARBOOK_PATH, 'mean_image.png')	
	imsave(mean_image_path, mean_img.astype(np.uint8))
	

if __name__ == "__main__":
	mean_image()
	pass
