from os import path
import random
import sys
import numpy as npf


SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')
YEARBOOK_PATH = path.join(DATA_PATH, 'yearbook')
YEARBOOK_TRAIN_PATH = path.join(YEARBOOK_PATH, 'train')
YEARBOOK_TRAIN_LABEL_PATH = path.join(YEARBOOK_PATH, 'yearbook_train.txt')
YEARBOOK_TRAIN_AUG_PATH = path.join(YEARBOOK_PATH, 'train_aug')
YEARBOOK_VALID_PATH = path.join(YEARBOOK_PATH, 'valid')
YEARBOOK_TRAIN_AUG_LABEL_PATH = path.join(YEARBOOK_PATH, 'yearbook_train_aug.txt')
YEARBOOK_VALID_LABEL_PATH = path.join(YEARBOOK_PATH, 'yearbook_valid.txt')

def read_labeled_image_list(image_list_file):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line.split()
        filenames.append(filename)
        if int(label) not in labels:
            labels.append(int(label))        
    shuffled_index = list(range(len(filenames)))
    shuffled_index_labels = list(range(len(labels)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index_labels]

    print('Found %d JPEG files across %d labels' %
        (len(filenames), len(labels)))
    return filenames, labels

def load_train():
  # Run it!
    image_list, label_list = read_labeled_image_list(YEARBOOK_TRAIN_AUG_LABEL_PATH)

load_train()
