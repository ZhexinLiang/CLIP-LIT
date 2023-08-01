import torch
import os
import time

import numpy as np
from PIL import Image
import glob
import time
import torchvision

def lowlight(image_path,image_list_path,result_list_path,DCE_net,size=256): 
	data_lowlight = Image.open(image_path)
	data_lowlight = data_lowlight.resize((size,size), Image.ANTIALIAS)
 

	data_lowlight = (np.asarray(data_lowlight)/255.0) 
	
	data_lowlight = torch.from_numpy(data_lowlight).float()
	
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0) 

	light_map = DCE_net(data_lowlight)

	enhanced_image = torch.clamp((data_lowlight / light_map), 0, 1)
	
	image_path = image_path.replace(image_list_path,result_list_path)
	
	image_path = image_path.replace('.JPG','.png')
	output_path = image_path
	if not os.path.exists(output_path.replace('/'+image_path.split("/")[-1],'')): 
		os.makedirs(output_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, output_path)

def inference(image_list_path,result_list_path,DCE_net,size=256):
    with torch.no_grad():
        filePath = image_list_path

        file_list = os.listdir(filePath)
        
        print("Inferencing...")
        for file_name in file_list:
            test_list = glob.glob(filePath+file_name)
            for image in test_list:
                lowlight(image,image_list_path,result_list_path,DCE_net,size)