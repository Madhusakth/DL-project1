from os import path
import util
import numpy as np
import argparse
from util import *
import csv
############
from skimage.io import imread, imsave
from skimage.transform import resize
from keras.models import load_model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import to_categorical
import tensorflow as tf

import time 
# Subtract mean, resize and rescale images here
def preprocess_img(image, mean_image=None, data_set='yearbook'):
	if data_set =='yearbook':
		new_image = np.pad(image,((7,8),(0,0),(0,0)), 'constant', constant_values=(0))
		new_image = np.array(new_image, dtype='float')
		if mean_image is not None:
			new_image = new_image - mean_image
		images_resized = resize(new_image,(256, 256, 3), mode='constant', anti_aliasing=True)
		new_image = images_resized / 255.0
		return new_image

	elif data_set == 'geolocation':
		return image

def load(image_path, mean_image=None, data_set='yearbook'):
	img=imread(image_path)
	img = preprocess_img(img, mean_image=mean_image, data_set=data_set)
	return img
	
class Predictor:
	DATASET_TYPE = 'yearbook'
	mean_image = None
	model=None
	dictionary={}
	
	# baseline 1 which calculates the median of the train data and return each time
	def yearbook_baseline(self):
		# Load all training data
		train_list = listYearbook(train=True, valid=False)

		# Get all the labels
		years = np.array([float(y[1]) for y in train_list])
		med = np.median(years, axis=0)
		return [med]

	# Compute the median.
	# We do this in the projective space of the map instead of longitude/latitude,
	# as France is almost flat and euclidean distances in the projective space are
	# close enough to spherical distances.
	def streetview_baseline(self):
		# Load all training data
		train_list = listStreetView(train=True, valid=False)

		# Get all the labels
		coord = np.array([(float(y[1]), float(y[2])) for y in train_list])
		xy = coordinateToXY(coord)
		med = np.median(xy, axis=0, keepdims=True)
		med_coord = np.squeeze(XYToCoordinate(med))
		return med_coord

	def set_mean_image(self):
		if 	self.DATASET_TYPE == 'yearbook':
			self.mean_image = imread(path.join(util.DATA_PATH, 'yearbook','mean_image.png'))
			self.mean_image = np.pad(self.mean_image,((7,8),(0,0),(0,0)), 'constant', constant_values=(0))
		elif self.DATASET_TYPE == 'geolocation':
			pass

	def load_trained_model(self, model_name=None):
		MODEL_PATH = path.join(util.SRC_PATH, '..', 'model')
		
		if model_name is None:
			if 	self.DATASET_TYPE == 'yearbook':
				model_file = 'vgg19_1_Classification.h5'
				dictionary_file = "yearbook_dict.npy"

			elif self.DATASET_TYPE == 'geolocation':
				model_file = 'vgg19_1_L1.h5' #change name
				dictionary_file = "geolocation_dict.npy"
		else:
			model_file = model_name
		# self.model = load_model(path.join(MODEL_PATH,model_file))
		self.model = self.yearbook_model(weights=path.join(MODEL_PATH,model_file))
		
		# Create dictionary to get back labels
		dict_np = np.load(path.join(MODEL_PATH,dictionary_file))
		for i,d in enumerate(dict_np):
			self.dictionary[i] = d
	
	def predict(self, image_path):
		s = time.time()
		
		# Load model for first time
		if self.model is None:
			self.set_mean_image()
			self.load_trained_model()
			print ("Loading model ..")

		# Load image and preprocess
		img = load(image_path, mean_image=self.mean_image, data_set=self.DATASET_TYPE)


		# Predict image
		img = np.expand_dims(img, axis=0)
		classes = self.model.predict(img)
		c = np.argmax(classes)
		k =time.time()

		print (" Time: ", k-s, ", Class: ", c, ", Actual class: ", self.dictionary[c], ", Prob",  classes[0,c])


		#TODO: predict model and return result either in geolocation format or yearbook format
		# depending on the dataset you are using
		if self.DATASET_TYPE == 'geolocation':
			result = self.streetview_baseline() #for geolocation
		elif self.DATASET_TYPE == 'yearbook':
			# result = self.yearbook_baseline() #for yearbook
			result = [self.dictionary[c]]
		return result
		
	
	def yearbook_model(self,weights=None):

		model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (256, 256, 3))

		# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
		#for layer in model.layers[:5]:
		#    layer.trainable = False

		#Adding custom Layers 
		x = model.output
		x = Flatten()(x)
		x = Dense(1024, activation="relu")(x)
		x = Dropout(0.5)(x)
		x = Dense(512, activation="relu")(x)
		x = Dense(104, activation=None)(x)
		predictions = Activation(tf.nn.softmax)(x)
		# predictions = Dense(1, activation=None)(x)

		# Creating the final model 
		model_final = Model(input = model.input, output = predictions)
		if weights is not None:
			model_final.load_weights(weights)
			print ("Weights loaded")
		model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])


		return model_final

