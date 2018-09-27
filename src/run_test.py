from os import path
import util
import numpy as np
import argparse
from skimage.io import imread
from util import *
from starter import load_train
import csv
#import vggrepo.vgg19_trainable_ours as vgg19
#import vggrepo.utils as utils

#from keras.applications.vgg16 import VGG16
#model = VGG16()
###################
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import to_categorical

###################
#def main():
#images,labels= load_train()
#print(images.shape)
#print(labels.shape)
#np.save('images_train.npy',images)
#np.save('labels_train.npy',labels)
#print("saved data")

images=np.load('images.npy')
labels=np.load('labels.npy')
print("loaded data successfully")
min_labels = np.min(labels)
max_labels = np.max(labels)

labels_scaled = labels - min_labels
one_hot_coded_labels = to_categorical(labels_scaled)
remove = np.array([1907,1917,1918,1920,1921]) - min_labels
one_labels_final = np.delete(one_hot_coded_labels,remove,axis=1)

print("one_labels_final size",one_labels_final.shape)
print("images shape",images.shape)
img_width, img_height = 171,186
#train_data_dir = "data/train"
#validation_data_dir = "data/val"
#?nb_train_samples = 4125
#?nb_validation_samples = 466 
batch_size = 16
epochs = 50


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(104, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model 
'''
model_final.fit_generator(
train_generator,
samples_per_epoch = 28000,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = nb_validation_samples,
callbacks = [checkpoint, early])
'''
model_final.fit(x = images,y=one_labels_final,batch_size=50,epochs=50,verbose=2,validation_split=0.3)

                                                                                                                                                                                          94,0-1        Bot
