import numpy as np
import os
import random
import sklearn
import sys
import time
import tensorflow as tf
import util
from skimage.io import imread
from skimage.transform import resize


########################
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import to_categorical
########################

# Network Parameters
img_width, img_height = 256, 256
batch_size = 16
epochs = 50
checkpoint_name="vgg19_1_L1"


# Load Data
train_images=np.load('train_aug_images.npy', mmap_mode='r')
train_labels=np.load('train_aug_labels.npy')
print("Loaded train data successfully")
print (train_images.shape)

valid_images=np.load('valid_images.npy', mmap_mode='r')
valid_labels=np.load('valid_labels.npy')
print("Loaded valid data successfully")

# Check model on small data set. See if it overfits
# k = 32
# train_images = train_images[:k]
# valid_images = valid_images[:k]
# train_labels = train_labels[:k]
# valid_labels = valid_labels[:k]


# Subtract mean, resize and rescale images here
mean_image = imread(os.path.join(util.DATA_PATH, 'yearbook','mean_image.png'))
mean_image = np.pad(mean_image,((7,8),(0,0),(0,0)), 'constant', constant_values=(0))

def preprocess_img(image_set):
    image_set = image_set - mean_image
    #print (image_set.shape)
    images_resized = np.array([resize(image,(img_width, img_height, 3), mode='constant', anti_aliasing=True) for image in image_set])
    #print (images_resized.shape)
    # Rescale the images
    image_set = images_resized / 255.0
    #print (image_set.shape)
    return image_set

print("one_labels_final size",train_labels.shape)
print("one_labels_final size",valid_labels.shape)
print("images shape",train_images.shape)

def generator(data_x, data_y, batch_size=32):
    num_samples = len(data_y)
    shuffled_index = list(range(num_samples))
    random.seed(12345)
    while 1:
        random.shuffle(shuffled_index)
        for offset in range(0, num_samples, batch_size):
            shuffled_batch = shuffled_index[offset:offset+batch_size]
            X_train = data_x[shuffled_batch]
            y_train = data_y[shuffled_batch]
            X_train = preprocess_img(X_train)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_images, train_labels, batch_size=batch_size)
valid_generator = generator(valid_images, valid_labels, batch_size=batch_size)


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
#for layer in model.layers[:5]:
#    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(1, activation=None)(x)

# Creating the final model 
model_final = Model(input = model.input, output = predictions)


# Load pre-trained weights
if os.path.isfile(checkpoint_name):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_name += timestr

# Load pre-trained weights
if len(sys.argv) >= 2 and os.path.isfile(sys.argv[1]):
    print ("Loading weights from ", sys.argv[1])
    model_final.load_weights(sys.argv[1])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_name = sys.argv[1][0:-3]+timestr

print ("Saving weights to ", checkpoint_name)

# compile the model 
model_final.compile(loss = "mean_absolute_error", optimizer = optimizers.Adam(lr=0.001), metrics=["mae"])


# Save the model according to the conditions  
checkpoint = ModelCheckpoint(checkpoint_name+'.h5', monitor='val_mean_absolute_error', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model 
model_final.fit_generator(train_generator, steps_per_epoch = len(train_images)/batch_size, validation_data=valid_generator, validation_steps=len(valid_labels)/batch_size, epochs=epochs, verbose=2, shuffle=True, callbacks = [checkpoint]) 
