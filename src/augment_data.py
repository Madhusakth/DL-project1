from __future__ import print_function
from os import path, listdir, unlink
import random
from skimage.io import imread, imsave
from skimage.util import random_noise
from skimage import img_as_ubyte
import warnings
from util import *


SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')
YEARBOOK_PATH = path.join(DATA_PATH, 'yearbook')
YEARBOOK_TRAIN_PATH = path.join(YEARBOOK_PATH, 'train')
YEARBOOK_TRAIN_LABEL_PATH = path.join(YEARBOOK_PATH, 'yearbook_train.txt')

YEARBOOK_TRAIN_AUG_PATH = path.join(YEARBOOK_PATH, 'train_aug')
YEARBOOK_TRAIN_AUG_LABEL_PATH = path.join(YEARBOOK_PATH, 'yearbook_train_aug.txt')

def noise_funciton(img):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		return img_as_ubyte(random_noise(img, mode='gaussian'))

def delete_files(folder_path):
	for the_file in listdir(folder_path):
		file_path = path.join(folder_path, the_file)
		try:
			if path.isfile(file_path):
				unlink(file_path)
			#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
			print(e)

def augment_data(flipped=True, noise=True, flip_thresh=0.15, noise_thresh=0.15):
	# TODO: Remove previous data and file
	delete_files(path.join(YEARBOOK_TRAIN_AUG_PATH, 'F'))
	delete_files(path.join(YEARBOOK_TRAIN_AUG_PATH, 'M'))

	# Load all training data
	train_list = listYearbook(train=True, valid=False)
	train_m_aug_list = []
	train_f_aug_list = []
	m = 0
	f = 0
	for image_gr_truth in train_list:
		if 'm' in image_gr_truth[0].lower():
			m += 1	
			train_m_aug_list.append(image_gr_truth)
		else:
			f += 1
			train_f_aug_list.append(image_gr_truth)

	# Randomly flip and add noise
	for image_gr_truth in train_list:
		image_path = path.join(YEARBOOK_TRAIN_PATH, image_gr_truth[0])
		flip_prob = random.uniform(0, 1)
		noise_prob = random.uniform(0, 1)
		flip_condition = flipped and (flip_prob <= flip_thresh)
		noise_condition = noise and (noise_prob <= noise_thresh)
		
		img=imread(image_path)
		image_aug_path = path.join(YEARBOOK_TRAIN_AUG_PATH, image_gr_truth[0])
		imsave(image_aug_path, img)		

		if not (flip_condition or noise_condition):
			continue

		if flip_condition:
			if 'm' in image_gr_truth[0].lower():
				m+=1
				new_image_path = 'M/'+str(m).zfill(6)+'.png'
				train_m_aug_list.append([new_image_path, image_gr_truth[1]])
			else:
				f+=1
				new_image_path = 'F/'+str(f).zfill(6)+'.png'
				train_f_aug_list.append([new_image_path, image_gr_truth[1]])
			image_aug_path = path.join(YEARBOOK_TRAIN_AUG_PATH, new_image_path)
			imsave(image_aug_path, img[:, ::-1])

		if noise_condition:
			noise_img = img.copy()
			noise_img = noise_funciton(noise_img)
			if 'm' in image_gr_truth[0].lower():
				m+=1
				new_image_path = 'M/'+str(m).zfill(6)+'.png'
				train_m_aug_list.append([new_image_path, image_gr_truth[1]])
			else:
				f+=1
				new_image_path = 'F/'+str(f).zfill(6)+'.png'
				train_f_aug_list.append([new_image_path, image_gr_truth[1]])
			image_aug_path = path.join(YEARBOOK_TRAIN_AUG_PATH, new_image_path)
			imsave(image_aug_path, noise_img)

	#Write the label files
	print ("Total Original : ", len(train_list))
	print ("Total Final : ", len(train_m_aug_list)+len(train_f_aug_list))
	
	f = open(YEARBOOK_TRAIN_AUG_LABEL_PATH, 'w')
	for image_gr_truth in train_m_aug_list:
		f.write('\t'.join(image_gr_truth)+'\n')
	for image_gr_truth in train_f_aug_list:
		f.write('\t'.join(image_gr_truth)+'\n')
	f.close()

if __name__ == "__main__":
	augment_data()
	pass
