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
checkpoint_name="vgg19_1_CE_MF"


# Load Data
train_images=np.load('train_aug_images.npy', mmap_mode='r')
train_labels=np.load('train_aug_labels.npy')
train_gender_labels=np.load('train_aug_gender_labels.npy')
print("Loaded train data successfully")
print (train_images.shape)

valid_images=np.load('valid_images.npy', mmap_mode='r')
valid_labels=np.load('valid_labels.npy')
valid_gender_labels=np.load('valid_gender_labels.npy')
print("Loaded valid data successfully")

# Convert Labels to one hot encoding vectors
unique_labels = np.unique(train_labels)
print ("Num Classes: ", len(unique_labels))
index_to_labels = {i:j for i,j in enumerate(unique_labels)}
labels_to_index = {j:i for i,j in enumerate(unique_labels)}

train_labels_encoded = [labels_to_index[label] for label in train_labels]
valid_labels_encoded = [labels_to_index[label] for label in valid_labels]

one_hot_train_labels = to_categorical(train_labels_encoded, num_classes=len(unique_labels))
one_hot_valid_labels = to_categorical(valid_labels_encoded, num_classes=len(unique_labels))

one_hot_train_gender_labels = to_categorical(train_gender_labels, num_classes=2)
one_hot_valid_gender_labels = to_categorical(valid_gender_labels, num_classes=2)


# # Check model on small data set. See if it overfits
# k = 32
# train_images = train_images[:k]
# valid_images = valid_images[:k]
# one_hot_train_labels = one_hot_train_labels[:k]
# one_hot_valid_labels = one_hot_valid_labels[:k]
# one_hot_train_gender_labels = one_hot_train_gender_labels[:k]
# one_hot_valid_gender_labels = one_hot_valid_gender_labels[:k]


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

print("one_labels_final size",one_hot_train_labels.shape)
print("one_gender_labels_final size",one_hot_train_gender_labels.shape)
print("images shape",train_images.shape)

def generator(data_x, data_y1, data_y2, batch_size=32):
    num_samples = len(data_y1)
    shuffled_index = list(range(num_samples))
    random.seed(12345)
    while 1:
        random.shuffle(shuffled_index)
        for offset in range(0, num_samples, batch_size):
            shuffled_batch = shuffled_index[offset:offset+batch_size]
            X_train = data_x[shuffled_batch]
            y1_train = data_y1[shuffled_batch]
            y2_train = data_y2[shuffled_batch]
            X_train = preprocess_img(X_train)
            y_train = {"classification": y1_train, "gender": y2_train}
            yield (X_train, y_train)

train_generator = generator(train_images, one_hot_train_labels, one_hot_train_gender_labels, batch_size=batch_size)
valid_generator = generator(valid_images, one_hot_valid_labels, one_hot_valid_gender_labels, batch_size=batch_size)


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
#for layer in model.layers[:5]:
#    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)

x1 = Dense(512, activation="relu")(x)
x1 = Dense(len(unique_labels), activation=None)(x1)
predictions_CE = Activation(tf.nn.softmax, name="classification")(x1)

x2 = Dense(512, activation="relu")(x)
x2 = Dense(2, activation=None)(x2)
predictions_MF = Activation(tf.nn.softmax, name="gender")(x2)

losses = {
    "classification": "categorical_crossentropy",
    "gender": "categorical_crossentropy"
}
# hyper-parameter
lossWeights = {"classification": 1.0, "gender": 0.5}

# Creating the final model 
model_final = Model(input = model.input, output = [predictions_CE, predictions_MF])


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

metrics = {
    "classification": "accuracy",
    "gender": "accuracy"
}

# compile the model 
model_final.compile(loss=losses, loss_weights=lossWeights, optimizer = optimizers.SGD(lr=0.001, momentum=0.9), metrics=metrics)


# Save the model according to the conditions  
checkpoint = ModelCheckpoint(checkpoint_name+'.h5', monitor='val_classification_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


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
# model_final.fit(x=train_images, y=one_hot_train_labels, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(valid_images, one_hot_valid_labels), callbacks = [checkpoint]) #callbacks = [checkpoint, early])
model_final.fit_generator(train_generator, steps_per_epoch = len(train_images)/batch_size, validation_data=valid_generator, validation_steps=len(valid_labels)/batch_size, epochs=epochs, verbose=2, shuffle=True, callbacks = [checkpoint]) 
