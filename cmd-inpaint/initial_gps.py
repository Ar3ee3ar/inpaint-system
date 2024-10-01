import cv2
import numpy as np

from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from tools.process_pano import Equirectangular,Perspective,inpaint_pano

width = 8192
height = 4096
mask_path = "mask/8k_focus_simple2.jpg"

# project mask image
# mask preprocess
# mask_img_full = preprocess_img(mask_path,img_size=(1000,1000)) # for inpaint in pano
mask_img = cv2.resize(cv2.imread(mask_path),(2000,2000)) # for model

# project to equirectangular (get mask image)
equ = Equirectangular(mask_img, 167, 27, -90,'img')    
mask_pano_img,_ = equ.GetEquirec(height,width) 
mask_pano_img = mask_pano_img.astype(np.uint8)
# ---------------------


# # per_img = preprocess_img(per_path,img_size=(1000,1000))
# mask_img_pano = cv2.imread(mask_path)
# mask_img_pano = cv2.resize(mask_img_pano,(2000,2000))
# # mask_img_pano = preprocess_img(mask_path,img_size=(1000,1000)) # for inpaint in pano
# # mask_img_crop = crop_center(mask_img_pano,(256,256))
# equ = Equirectangular(mask_img_pano,167, 27, -90,'img')    
# mask_pano_img,_ = equ.GetEquirec(4096,8192)
# mask_pano_img = mask_pano_img.astype(np.uint8)