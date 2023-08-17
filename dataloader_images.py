import os
import sys

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random
import cv2
import clip

random.seed(1143)

def transform_matrix_offset_center(matrix, x, y):
	"""Return transform matrix offset center.

	Parameters
	----------
	matrix : numpy array
		Transform matrix
	x, y : int
		Size of image.

	Examples
	--------
	- See ``rotation``, ``shear``, ``zoom``.
	"""
	o_x = float(x) / 2 + 0.5
	o_y = float(y) / 2 + 0.5
	offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
	reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
	transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
	return transform_matrix 

def img_rotate(img, angle, center=None, scale=1.0):
	"""Rotate image.
	Args:
		img (ndarray): Image to be rotated.
		angle (float): Rotation angle in degrees. Positive values mean
			counter-clockwise rotation.
		center (tuple[int]): Rotation center. If the center is None,
			initialize it as the center of the image. Default: None.
		scale (float): Isotropic scale factor. Default: 1.0.
	"""
	(h, w) = img.shape[:2]

	if center is None:
		center = (w // 2, h // 2)

	matrix = cv2.getRotationMatrix2D(center, angle, scale)
	rotated_img = cv2.warpAffine(img, matrix, (w, h),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0,0,0),)
	return rotated_img

def zoom(x, zx, zy, row_axis=0, col_axis=1):
	zoom_matrix = np.array([[zx, 0, 0],
							[0, zy, 0],
							[0, 0, 1]])
	h, w = x.shape[row_axis], x.shape[col_axis]

	matrix = transform_matrix_offset_center(zoom_matrix, h, w) 
	x = cv2.warpAffine(x, matrix[:2, :], (w, h),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0,0,0),)
	return x

def augmentation(img1,img2):
	hflip=random.random() < 0.5
	vflip=random.random() < 0.5
	rot90=random.random() < 0.5
	rot=random.random() <0.3
	zo=random.random()<0.3
	angle=random.random()*180-90
	if hflip:
		img1=cv2.flip(img1,1)
		img2=cv2.flip(img2,1)
	if vflip:
		img1=cv2.flip(img1,0)
		img2=cv2.flip(img2,0)
	if rot90:
		img1 = img1.transpose(1, 0, 2)
		img2 = img2.transpose(1,0,2)
	if zo:
		zoom_range=(0.7, 1.3)
		zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
		img1=zoom(img1, zx, zy)
		img2=zoom(img2,zx,zy)
	if rot:
		img1=img_rotate(img1,angle)
		img2=img_rotate(img2,angle)
	return img1,img2

def preprocess_aug(img1,img2):
	img1 = np.uint8((np.asarray(img1)))
	img2 = np.uint8((np.asarray(img2)))
	img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
	img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
	img1,img2=augmentation(img1,img2)
	img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
	img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
	return img1,img2

device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
#load clip

model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model/")#ViT-B/32
for para in model.parameters():
	para.requires_grad = False

def populate_train_list(lowlight_images_path,overlight_images_path=None):
	if overlight_images_path!=None:
		image_list_lowlight = glob.glob(lowlight_images_path + "*")
		image_list_overlight = glob.glob(overlight_images_path + "*")
		image_list_lowlight += image_list_overlight
	else:
		image_list_lowlight = glob.glob(lowlight_images_path + "*")

	train_list = sorted(image_list_lowlight)
	#print(train_list)
	random.shuffle(train_list)

	return train_list

	

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path,overlight_images_path=None):

		self.train_list = populate_train_list(lowlight_images_path,overlight_images_path) 
		self.size = 512

		self.data_list = self.train_list
		print("Total training examples (Backlit):", len(self.train_list))


	def __getitem__(self, index):

		data_lowlight_path = self.data_list[index]
		data_lowlight = Image.open(data_lowlight_path)

		if("result" not in data_lowlight_path):
			data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		data_lowlight,_=preprocess_aug(data_lowlight,data_lowlight)
		
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight_output = torch.from_numpy(data_lowlight).float().permute(2,0,1)
		
		return data_lowlight_output,data_lowlight_path

	def __len__(self):
		return len(self.data_list)

