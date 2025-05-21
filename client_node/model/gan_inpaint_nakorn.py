import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import os
import time
# import wandb
# import matplotlib.pyplot as plt
from distutils.util import strtobool
from PIL import Image
import time
import glob
import io
import base64

from .dataset import Dataset, createAugment
# pix2pix
from .pix2pix import Generator as p2pG
from .pix2pix import Discriminator as p2pD
# tool
# from .tools.utils import generate_images, wandb_log, view_test
# from .tools.loss import discriminator_loss,generator_loss
from .tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from .tools.process_pano import Equirectangular,Perspective,inpaint_pano
from .config_gan_nakorn import image_size,width,height,in_mask,generator, mask_pano_img, mask_img_full,mask_img_seamless


tf.config.run_functions_eagerly(True)

# Description: command ไว้ทำตามคำสั่ง
# def _argparse():
#     # print('parsing args...')
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img', '-loss-list', nargs='+', default=[], help = "number of image")
#     parser.add_argument("--mask", "-mask_dir",type=str, default='', help="directory that store mask")  
#     parser.add_argument("--weightG", "-model_weight_g",type=str, default='', help="path to generator model weight")
#     parser.add_argument("--model_name",type=str, default='p2p', help="model name : p2p | pconv")
#     parser.add_argument("--random_mask", type=str, nargs='?', const='True', default="False", help="enable random mask")
#     parser.add_argument("--save", type=str, nargs='?', const='True', default="False", help="enable random mask")
#     parser.add_argument("--save_fol",type=str, default='pano', help="folder name")
#     arg = parser.parse_args()
#     return arg

def seamless(source,target):
    mask = mask_img_seamless
    center = (499,450)
    im_clone = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
    return im_clone

def save_img(path, inpaint_img):
    # save inpaint image
    print('save')
    impainted_image_im = Image.fromarray(((inpaint_img)*255.0).astype(np.uint8))
    impainted_image_im.save(path)


def main_gan_nakorn(main_path,save_path,target_coor):
    # try:
    # for i,dir_input in enumerate(dir_list_input):
    all_time = 0.0
    img_path = main_path + "img_"+str(target_coor).rjust(6, '0')+".jpg"
    # print(img_path)
    start_pano = time.time()

    # load image with preprocess
    car_pano_img = preprocess_img(img_path,img_size=(width,height))

    # project to perspective (get car image)
    start_proj = time.time()
    equ = Perspective(car_pano_img)    # Load equirectangular image
    per_img_1000 = equ.GetPerspective(167, 27, -90, 1000, 1000)  # Specify parameters(FOV, theta, phi, height, width)
    end_proj = time.time()
    print('proj: ',end_proj - start_proj)

    # car image
    per_img = crop_center(per_img_1000,(512,512)) # resize to 290,190
    car_img = preprocess_img(per_img,img_size=image_size) # resize to match input model (256,256)

    car_img = car_img.astype('float32')


    testgen = createAugment(np.array([car_img]), np.array([car_img]),np.array([in_mask]),batch_size=1,dim=image_size, shuffle=False, random_mask=False)
    [masked_image, mask], car_img = testgen[0]
    # plt.imshow(masked_image[0])
    # plt.show()

    # pix2pix ------------------------------------
    car_img = (car_img - 0.5)/0.5 # normalize [-1,1]
    masked_image = (masked_image - 0.5)/0.5 # normalize [-1,1]
    inputs = masked_image # pix2pix
    start_model = time.time()
    predict_image = generator(inputs, training=True)
    end_model = time.time()
    print('model inpaint time: ',(end_model - start_model))
    predict_image_norm = ((predict_image[0]* 0.5) + 0.5).numpy()# de-normalize [0,1]
    # plt.imshow(predict_image_norm)
    # plt.show()
    masked_image = (masked_image* 0.5) + 0.5 # de-normalize [0,1]
    car_img = (car_img* 0.5) + 0.5 # de-normalize [0,1]
    # ------------------------------------------------

    mask_img_pano = mask_img_full/255.0
    per_img_1000 = per_img_1000/255.0

    start_pro_img = time.time()
    predict_image_norm = preprocess_img(predict_image_norm,img_size=(512,512)) # resize to match input model (256,256)
    predict_image_norm = pad_images_to_same_size(predict_image_norm)
    # predict_image_norm = ((1 - mask_img_pano) * per_img_1000) + (mask_img_pano * predict_image_norm)
    predict_image_norm = seamless((predict_image_norm*255.0).astype(np.uint8),(per_img_1000*255.0).astype(np.uint8))
    predict_image_norm = predict_image_norm/255.0
    end_pro_img = time.time()
    print('overlay: ',end_pro_img - start_pro_img)

    start_reproj = time.time()
    equ = Equirectangular(predict_image_norm,167, 27, -90)
    inpaint_pano_img,_ = equ.GetEquirec(height,width)
    end_reproj = time.time()
    print('reproj: ',(end_reproj - start_reproj)) 

    inpaint_fill = inpaint_pano(inpaint_pano_img*255.0,mask_pano_img,car_pano_img)
    # inpaint_mask = inpaint_pano(mask_pano_img,mask_pano_img,car_pano_img)
    end_pano = time.time()

    # if(output_mode=="save"):
    save_path = save_path +  "img_"+str(target_coor).rjust(6, '0')+"_O.jpg"
    print(save_path)
    start_save = time.time()
    save_img(path=save_path,inpaint_img=inpaint_fill)
    end_save = time.time()
    print('save_time: ',(end_save - start_save))
    all_time = all_time + (end_pano - start_pano)
    print('all_time: ',all_time)
    # return {"message": "Success"}
    # elif(output_mode=="export"):
    #     # print(inpaint_fill.shape)
    #     # print(type(inpaint_fill))
    #     pil_img = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
    #     img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    #     res, im_png = cv2.imencode(".jpg", img)
    #     encoded_img = base64.b64encode(im_png)
    #     return {"message": "Success","img": encoded_img}
            
    # except Exception as error:
    #     return {"message": error}
    


# if __name__ == '__main__':
#     mms_fol = "C:/Users/acer/Desktop/input/"
#     save_fol = "C:/Users/acer/Desktop/output/"
#     main(mms_fol, save_fol)

