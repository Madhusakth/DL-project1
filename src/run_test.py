import numpy as np
import os
import random
import tensorflow as tf
import util
from skimage.io import imread
from skimage.transform import resize
import sklearn

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

# Load Data
train_images=np.load('train_aug_images.npy')
train_labels=np.load('train_aug_labels.npy')
print("Loaded train data successfully")
print (train_images.shape)

valid_images=np.load('valid_images.npy')
valid_labels=np.load('valid_labels.npy')
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

# # Check model on small data set. See if it overfits
# k = 128
# train_images = train_images[:k]
# valid_images = valid_images[:k]
# one_hot_train_labels = one_hot_train_labels[:k]
# one_hot_valid_labels = one_hot_valid_labels[:k]

# Subtract mean, resize and rescale images here
mean_image = imread(os.path.join(util.DATA_PATH, 'yearbook','mean_image.png'))
mean_image = np.pad(mean_image,((7,8),(0,0),(0,0)), 'constant', constant_values=(0))
train_images = train_images - mean_image
valid_images = valid_images - mean_image
print (train_images.shape)

train_images_resized = np.array([resize(image,(img_width, img_height, 3), mode='constant', anti_aliasing=True) for image in train_images])
valid_images_resized = np.array([resize(image,(img_width, img_height, 3), mode='constant', anti_aliasing=True) for image in valid_images])

print (train_images_resized.shape)

# Rescale the images
train_images = train_images_resized / 255.0
valid_images = valid_images_resized / 255.0

print (train_images.shape)

print("one_labels_final size",one_hot_train_labels.shape)
print("one_labels_final size",one_hot_valid_labels.shape)
print("images shape",train_images.shape)

def generator(data_x, data_y, batch_size=32):
    num_samples = len(data_x)
    shuffled_index = list(range(num_samples))
    random.seed(12345)
    while 1:
        random.shuffle(shuffled_index)
        data_x = data_x[shuffled_index]
        data_y = data_y[shuffled_index]
        for offset in range(0, num_samples, batch_size):
            X_train = np.array(data_x[offset:offset+batch_size])
            y_train = np.array(data_y[offset:offset+batch_size])
            
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_images, one_hot_train_labels, batch_size=batch_size)
valid_generator = generator(valid_images, one_hot_valid_labels, batch_size=batch_size)


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
#for layer in model.layers[:5]:
#    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dense(len(unique_labels), activation=None)(x)
predictions = Activation(tf.nn.softmax)(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg19_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
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
