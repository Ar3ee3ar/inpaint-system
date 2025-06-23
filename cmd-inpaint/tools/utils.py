import sys
import os
import glob
import cv2
import numpy as np
# from tensorflow import keras
# import tensorflow as tf
# import inspect
# check corrupted image
from skimage import io

from .process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
# from .pix2pix import Generator as p2pG
# from .pix2pix import Discriminator as p2pD
# from .deepfill.inpaint_model import InpaintGenerator
# from .sam_tools.segment_anything import sam_model_registry

# ------------------ function check corrupt file ------------------
def write_file(path,msg):
    with open(path, "a", encoding="utf-8") as f:
        # if(str(e).find('image file is truncated') != -1):
        # error_msg = "error: "+path.replace('\\','/')+" ไฟล์เสียหาย (corrupted image)"
        f.write(msg+'\n')

def check_corrupt(path,report_file=''):
    # check  binary file -----------------------
    # with open(path, 'rb') as f:
    #     check_chars = f.read()[-2:]
    # if check_chars != b'\xff\xd9':
    #     # print('Not complete image')
    #     error_msg = "error: "+path.replace('\\','/')+" ไฟล์เสียหาย (corrupted image)"
    #     print(error_msg)
    #     write_file(report_file, error_msg)
    #     return True
    # else:
    #     return False
    # check with skimage -------------------------
    try:
        _ = io.imread(path)
        return False
        # Do stuff with img
    except Exception as e:
        # print('found error ',e)
        # if(not os.path.exists(report_file)): # if not found output folder -> create folder
        #     create_fol(save_path)
        # with open(report_file, "a", encoding="utf-8") as f:
        if(str(e).find('image file is truncated') != -1):
            error_msg = "error: "+path.replace('\\','/')+" ไฟล์เสียหาย (corrupted image)"
            if(report_file!=''):
                write_file(report_file, error_msg)
                # f.write(error_msg+'\n')
            print(error_msg)
        return True

# -------------- utility tools --------------------------------------
def create_fol(path):
    if not os.path.exists(path): 
        os.makedirs(path)
    else:
        print('already create '+path)

def find_fol(main_input):
    dir_list_input = glob.glob(main_input+'/**/'+'Camera/',recursive=True)
    for i in range(len(dir_list_input)):
        dir_list_input[i] = dir_list_input[i].replace('/','/')
    return dir_list_input

def gen_tranparent_mask(src):
    # Convert image to image gray 
    # print(src.shape)
    if(len(src.shape)==3):
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 
        
        # Applying thresholding technique 
        _, alpha = cv2.threshold(tmp, thresh=127, maxval=255, type=cv2.THRESH_BINARY) 
        
        # Using cv2.split() to split channels  
        # of coloured image 
        b, g, r = cv2.split(src) 
        
        # Making list of Red, Green, Blue 
        # Channels and alpha 
        rgba = [b, g, r, alpha] 
        
        # Using cv2.merge() to merge rgba 
        # into a coloured/multi-channeled image 
        dst = cv2.merge(rgba, 4) 
        # Writing and saving to a new image 
        #cv2.imwrite(save_path+count+"_gfg_white.png", dst)
    elif(len(src.shape)==2):
        # Convert to numpy array
        gray_np = np.array(src)

        # Create an RGBA image: black with alpha from grayscale
        rgba_np = np.zeros((gray_np.shape[0], gray_np.shape[1], 4), dtype=np.uint8)
        rgba_np[..., 3] = gray_np  # set alpha channel 
        dst = rgba_np
    # plt.imshow(dst)
    # plt.show()
    return dst

def binary_change(mask,save_path=''):
    # Apply threshold to make it binary (0 and 1)
    binary_mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    # print(binary_mask)
    # cv2.imwrite(save_path,binary_mask)
    # plt.imshow(binary_mask)
    # plt.show()
    return binary_mask

def up_downscale(img,img_size,type_scale,type_process='resize'):
    if(type_process=='pad_image'):
        if(type_scale == 'upscale'):
            result_img = pad_images_to_same_size(img,img_size[0],img_size[1])
        elif(type_scale == 'downscale'):
            result_img = crop_center(img,img_size) # resize to 290,190
    elif(type_process=='resize'):
        if(type_scale == 'upscale'):
            # result_img = main_upscale(img)
            result_img = cv2.resize(img,img_size)
        elif(type_scale == 'downscale'):
            result_img = cv2.resize(img,img_size)
    return(result_img)
# ------------ iterate through dictionary ------------------------
class DictNavigator:
    def __init__(self, data_dict):
        self.data = data_dict
        self.keys = list(data_dict.keys())
        self.index = 0  # Start at the first item

    def current(self):
        key = self.keys[self.index]
        return key, self.data[key]

    def next(self):
        if self.index < len(self.keys) - 1:
            self.index += 1
        return self.current()

    def previous(self):
        if self.index > 0:
            self.index -= 1
        return self.current()