from ctypes import sizeof
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



def augmentation(img,hflip,vflip,rot90,rot,zo,angle,zx,zy):
    if hflip:
        img=cv2.flip(img,1)
    if vflip:
        img=cv2.flip(img,0)
    if rot90:
        img = img.transpose(1, 0, 2)
    if zo:
        img=zoom(img, zx, zy)
    if rot:
        img=img_rotate(img,angle)
    return img

def preprocess_aug(img_list):
    hflip=random.random() < 0.5
    vflip=random.random() < 0.5
    rot90=random.random() < 0.5
    rot=random.random() <0.3
    zo=random.random()<0.3
    angle=random.random()*180-90
    zoom_range=(0.5, 1.5)
    zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    aug_img_list=[]
    for img in img_list:
        img = np.uint8((np.asarray(img)))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img=augmentation(img,hflip,vflip,rot90,rot,zo,angle,zx,zy)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        aug_img_list.append(img)
    return aug_img_list

device = "cpu"
#load clip
model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model/")#ViT-B/32
for para in model.parameters():
    para.requires_grad = False

def populate_train_list(lowlight_images_path,normallight_images_path=None,overlight_images_path=None):
    image_list_lowlight = glob.glob(lowlight_images_path + "*")
    image_list_normallight = glob.glob(normallight_images_path+"*")
    
    image_ref_list=image_list_normallight.copy()
    image_input_list=image_list_lowlight.copy()
    if len(image_list_normallight)==0 or len(image_list_lowlight)==0:
        raise Exception("one of the image lists is empty!", len(image_list_normallight),len(image_list_lowlight))
    if len(image_list_normallight)<len(image_list_lowlight):
            while(len(image_ref_list)<len(image_list_lowlight)):
                for i in image_list_normallight:
                    image_ref_list.append(i)
                    if(len(image_ref_list)>=len(image_list_lowlight)):
                        break
    else:
        while(len(image_input_list)<len(image_list_normallight)):
            for i in image_list_lowlight:
                image_input_list.append(i)
                if(len(image_input_list)>=len(image_list_normallight)):
                    break
    train_list1=image_input_list
    train_list2=image_ref_list
    # print(train_list1)
    random.shuffle(train_list1)
    random.shuffle(train_list2)

    return train_list1,train_list2

def preprocess_feature(img):
    img = (np.asarray(img)/255.0) 
    img = torch.from_numpy(img).float()
    img=img.permute(2,0,1).to(device)
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_resize = transforms.Resize((224,224))
    img=img_resize(img)
    img=clip_normalizer(img.reshape(1,3,224,224))
    image_features = model.encode_image(img)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path,normallight_images_path,semi1_path=None,semi2_path=None):
        self.train_list1,self.train_list2 = populate_train_list(lowlight_images_path,normallight_images_path)
        self.size = 256
        self.neg_path=lowlight_images_path
        self.semi1_path=semi1_path
        self.semi2_path=semi2_path
        self.data_list = self.train_list1
        print("Total training examples (Well-lit):", len(self.train_list2))
        

    def __getitem__(self, index):

        data_lowlight_path = self.data_list[index]
        ref_path = self.train_list2[index]
        
        data_lowlight = Image.open(data_lowlight_path)
        ref = Image.open(ref_path)

        data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
        ref = ref.resize((self.size,self.size), Image.ANTIALIAS)
        if self.semi1_path==None:
            img_list=preprocess_aug([data_lowlight,ref])
        elif self.semi2_path==None:
            semi1 = Image.open(data_lowlight_path.replace(self.neg_path,self.semi1_path).replace('.JPG','.png'))
            img_list=preprocess_aug([data_lowlight,semi1,ref])
        else:
            semi1 = Image.open(data_lowlight_path.replace(self.neg_path,self.semi1_path).replace('.JPG','.png'))
            semi2 = Image.open(data_lowlight_path.replace(self.neg_path,self.semi2_path).replace('.JPG','.png'))
            img_list=preprocess_aug([data_lowlight,semi1,semi2,ref])
            
        img_feature_list=[]
        for img in img_list:
            img_feature=preprocess_feature(img)
            img_feature_list.append(img_feature)
        
        return img_feature_list,1

    def __len__(self):
        return len(self.data_list)

