from tensorflow import keras
import cv2

from .pix2pix import Generator as p2pG
from .tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from .tools.process_pano import Equirectangular,Perspective,inpaint_pano

im_size = 512
image_size = (im_size,im_size)
input_model_size = [im_size,im_size,3]

width = 4096
height = 2048
mask_path = "./model/mask/binary_mask_test.jpg"
mask_img_seamless_path = "./model/mask/8k_focus_simple2.jpg"

# project mask image
# mask preprocess
mask_img_full = preprocess_img(mask_path,img_size=(1000,1000)) # for inpaint in pano
mask_img_full[mask_img_full < 250] = 0
mask_img = preprocess_img(mask_path,img_size=(1000,1000)) # for model
mask_img[mask_img < 250] = 0
# mask image
mask_img_crop = crop_center(mask_img,(512,512)) # resize to 290,190
mask_img = preprocess_img(mask_img_crop,img_size=image_size) # resize to match input model (256,256)
# seamless mask
mask_img_seamless = cv2.resize(cv2.imread(mask_img_seamless_path),(1000,1000)) # for model
# project to equirectangular (get mask image)
equ = Equirectangular(mask_img_full,167, 27, -90,'img')    
mask_pano_img,_ = equ.GetEquirec(height,width) 
mask_bw = cv2.bitwise_not(mask_img) # for only u-net
in_mask = mask_bw.astype('float32')
# ---------------------

# initial generator model
keras.backend.clear_session()
generator = p2pG.Generator(input_shape=input_model_size)
generator.load_weights("./model/weight/bestG_weight_nakorn.h5") # 500