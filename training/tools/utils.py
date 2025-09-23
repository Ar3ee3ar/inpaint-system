import sys
import os
import glob
import cv2
import wandb
import numpy as np
from typing import Union
from dotenv import load_dotenv
# check corrupted image
from skimage import io

from .process_img import crop_center, pad_images_to_same_size
sys.path.append("..")
# from tools.Real_ESRGAN.inference_realesrgan_func import main_upscale
# ----------- log in wandb -----------------
def wandb_log(cfg):
  dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
  load_dotenv(dotenv_path)
  print(os.environ.get('WANDB_KEYS'),' ',os.environ.get('WANDB_ENTITY'),' ',os.environ.get('WANDB_PROJECT'))
  wandb.login(key=os.environ.get('WANDB_KEYS'))
  # initial wandb log
  wandb.tensorboard.patch(root_logdir="")
  wandb.init(entity=os.environ.get('WANDB_ENTITY'), project=os.environ.get('WANDB_PROJECT'), sync_tensorboard=True, config=cfg)
  return wandb.run.dir
# ------------------ function check corrupt file ------------------
def write_file(path:str,msg:str) -> None:
    """
    Appends a message to a file at the specified path.
    Args:
        path (str): The file path where the message will be written.
        msg (str): The message to append to the file.
    Returns:
        None
    Notes:
        - The message is appended to the file with a newline character.
        - The file is opened in append mode with UTF-8 encoding.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg+'\n')

def check_corrupt(path:str,report_file:str='') -> bool:
    """
    Check corrupt image file
    Args:
        path (str): The image file path for checking corruption
        report_file (str): The file path where the message will be written.
    Returns:
        (boolean): if image corrupt or not
    Notes:
        - If image is corrupt return true
        - If image is not corrupt return false
    """
    # (Alternative method to check corrupt file )check binary file -----------------------
    # with open(path, 'rb') as f: # Open the file at 'path' in binary read mode
    #     check_chars = f.read()[-2:] # Read the entire file and get the last two bytes
    # if check_chars != b'\xff\xd9': # Check if the last two bytes are not the JPEG end marker
    #     # print('Not complete image')
    #     error_msg = "error: "+path.replace('\\','/')+" ไฟล์เสียหาย (corrupted image)" # Create an error message
    #     print(error_msg) # Print the error message
    #     write_file(report_file, error_msg) # Write the error message to a report file
    #     return True # Return True to indicate the file is corrupted
    # else:
    #     return False # Return False to indicate the file is not corrupted
    # check with skimage -------------------------
    try:
        _ = io.imread(path) # use skimage.io library to write path
        return False # if not show error indicate that image is not corrupt
        # Do stuff with img
    except Exception as e:
        if(str(e).find('image file is truncated') != -1): # if error show 'image file is truncated'
            error_msg = "error: "+path.replace('\\','/')+" ไฟล์เสียหาย (corrupted image)" # create error message
            if(report_file!=''): # if report_file not found -> no need to write error to report file
                write_file(report_file, error_msg) # if report_file is found then call write_file function to write error message
            print(error_msg) # show error message on comand line
        return True # Return True to indicate the file is corrupted

# -------------- utility tools --------------------------------------
def create_fol(path:str) -> None:
    """
    Creates a directory at the specified path if it does not already exist.
    Args:
        path (str): The path of the directory to create.
    Returns:
        None
    Side Effects:
        - Creates the directory (and any necessary parent directories) if it does not exist.
        - Prints a message if the directory already exists.
    """

    if not os.path.exists(path): 
        os.makedirs(path)
    else:
        print('already create '+path)

def find_fol(main_input:str, stored_folder:str = "Camera") -> list:
    """
    Searches for all directories named 'Camera' within the given main_input directory and its subdirectories.
    Args:
        main_input (str): The root directory path in which to search for 'Camera' directories.
        stored_data (str): The format of folder that stored image (default = 'Camera')
    Returns:
        list: A list of paths to all found 'Camera' directories under main_input.
    """
    dir_list_input = glob.glob(main_input+'/**/'+stored_folder+'/',recursive=True)
    for i in range(len(dir_list_input)):
        dir_list_input[i] = dir_list_input[i].replace('/','/')
    return dir_list_input

def gen_tranparent_mask(src:np.ndarray) -> np.ndarray:
    """
    Generates an RGBA image with a transparency (alpha) mask based on the input image.
    For color images (3 channels), the function converts the image to grayscale,
    applies a binary threshold to create an alpha channel, and merges it with the original
    BGR channels to produce an RGBA image.
    For grayscale images (2 channels), the function creates an RGBA image where the alpha
    channel is set from the grayscale values and the RGB channels are set to zero (black).
    Args:
        src (numpy.ndarray): Input image. Can be a 2D grayscale or 3D BGR image.
    Returns:
        numpy.ndarray: Output RGBA image with the alpha channel representing transparency.
    """

    # Convert image to image gray 
    if(len(src.shape)==3):
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 
        # Applying thresholding technique 
        _, alpha = cv2.threshold(tmp, thresh=127, maxval=255, type=cv2.THRESH_BINARY) 
        # Using cv2.split() to split channels of coloured image 
        b, g, r = cv2.split(src) 
        # Making list of Red, Green, Blue Channels and alpha 
        rgba = [b, g, r, alpha] 
        # Using cv2.merge() to merge rgba into a coloured/multi-channeled image 
        dst = cv2.merge(rgba, 4) 
        # Writing and saving to a new image 
        # cv2.imwrite(save_path+count+"_gfg_white.png", dst)
    elif(len(src.shape)==2):
        # Convert to numpy array
        gray_np = np.array(src)
        # Create an RGBA image: black with alpha from grayscale
        rgba_np = np.zeros((gray_np.shape[0], gray_np.shape[1], 4), dtype=np.uint8)
        rgba_np[..., 3] = gray_np  # set alpha channel 
        dst = rgba_np
    return dst

def binary_change(mask:np.ndarray, save_path:str='') -> np.ndarray:
    """
    Converts a grayscale mask image to a binary mask using a threshold.
    Parameters:
        mask (np.ndarray): Input grayscale mask image as a NumPy array.
        save_path (str, optional): Path to save the binary mask image. If empty, the image is not saved. Default is ''.
    Returns:
        np.ndarray: Binary mask image where pixel values are either 0 or 255.
    Notes:
        Can use in case of mask is not purely in 0 or 255
    """
    
    # Apply threshold to make it binary (0 and 1)
    binary_mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    if save_path != '':
      cv2.imwrite(save_path,binary_mask)
      
    return binary_mask

def up_downscale(img:np.ndarray, img_size:Union[list,tuple], type_scale:str, type_process:str='resize') -> np.ndarray:
    """
    upscale(upsize) or downscale(downsize) image
    Args:
        img (np.ndarray): Numpy image
        img_size (list | tuple): desired output size of image
        type_scale (str): type of scale method
            'upscale' for increase size of image
            'downscale' for decrease size of image
        type_process (str): overall inpainting process
            'pad_image' for [crop (to match model input size), padding (to match original perspective size)] process
            'resize' for [downsize (to match model input size), upsize (to match original perspective size)] process
    Return:
        np.ndarray: image where size match img_size
    """
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
    """
    A utility class for navigating through the key-value pairs of a dictionary in a sequential manner.
    Attributes:
        data (dict): The dictionary to navigate.
        keys (list): List of keys from the dictionary, preserving insertion order.
        index (int): The current position in the keys list.
    """
    def __init__(self, data_dict:dict) -> None:
        """
        Initializes the DictNavigator with the provided dictionary.
        """
        self.data = data_dict
        self.keys = list(data_dict.keys())
        self.index = 0  # Start at the first item

    def current(self):
        """
        Returns the current key and its corresponding value as a tuple.
        """
        key = self.keys[self.index]
        return key, self.data[key]

    def next(self):
        """
        Moves to the next key-value pair (if available) and returns it.
        """
        if self.index < len(self.keys) - 1:
            self.index += 1
        return self.current()

    def previous(self):
        """
        Moves to the previous key-value pair (if available) and returns it.
        """
        if self.index > 0:
            self.index -= 1
        return self.current()