import sys
import os
import glob
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
import inspect

from .process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
# from .pix2pix import Generator as p2pG
# from .pix2pix import Discriminator as p2pD
# from .deepfill.inpaint_model import InpaintGenerator
# from .sam_tools.segment_anything import sam_model_registry

# -------------- utility tools --------------------------------------

# Helper to get file path near the .exe
def get_file_path(filename, base_path=None, must_exist=True):
    """
    Locate a file based on caller's path or specified base path, with _MEIPASS fallback.

    Args:
        filename (str): File name to locate.
        base_path (str, optional): Base directory to search from. If None, use caller's path.
        must_exist (bool): If True, raise if not found.

    Returns:
        str: Full path to the file.

    Raises:
        FileNotFoundError: If not found and must_exist is True.
    """
    # Use caller’s path if base_path not provided
    if base_path is None:
        caller_path = os.path.abspath(os.path.dirname(inspect.stack()[1].filename))
        base_path = caller_path

    # Check local path
    local_path = os.path.join(base_path, filename)
    if os.path.exists(local_path):
        return local_path

    # Fallback to _MEIPASS if available
    if hasattr(sys, "_MEIPASS"):
        # Preserve relative layout inside bundle
        relative_path = os.path.relpath(base_path, start=os.path.dirname(sys.argv[0]))
        meipass_path = os.path.join(sys._MEIPASS, relative_path, filename)
        if os.path.exists(meipass_path):
            return meipass_path

    if must_exist:
        raise FileNotFoundError(f"File '{filename}' not found in '{base_path}' or _MEIPASS.")
    return local_path

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