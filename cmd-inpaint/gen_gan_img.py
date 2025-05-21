import argparse
from tensorflow import keras
import tensorflow as tf
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import utm
import scipy.sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import glob
from distutils.util import strtobool
from dataset import Dataset, createAugment
# pix2pix
from pix2pix import Generator as p2pG
from pix2pix import Discriminator as p2pD
# deepfill
from deepfill.inpaint_model import InpaintGenerator
from PIL import Image
# import yaml
# tool
from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from tools.process_pano import Equirectangular,Perspective,inpaint_pano
# from initial_gps import *
import os
import sys
import glob
import time
import threading
import json

import matplotlib.pyplot as plt

# degree สะพาน (167, 0, -90, 1000, 1000)
# degree สะพาน (167, 27, -80, 1000, 1000)

def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-main_dir",type=str, default='', help="parent directory that store dataset")
    parser.add_argument("--overwrite", "-overwrite", type=str, nargs='?', const='True', default="False", help="enable wandb log")
    parser.add_argument("--config", "-config",type=str, default='', help="config cars")
    arg = parser.parse_args()
    return arg

# Helper to get file path near the .exe
def get_file_path(filename):
    """
    Try to locate the file either in the bundled PyInstaller directory (_MEIPASS)
    or externally next to the executable or script.
    """
    paths_to_try = []

    # If running from PyInstaller
    if getattr(sys, 'frozen', False):
        # 1. Try bundled (sys._MEIPASS)
        if hasattr(sys, '_MEIPASS'):
            paths_to_try.append(os.path.join(sys._MEIPASS, filename))
        # 2. Try external (next to the .exe)
        paths_to_try.append(os.path.join(os.path.dirname(sys.executable), filename))
    else:
        # 3. When running from source (.py file)
        base_path = os.path.dirname(os.path.abspath(__file__))
        paths_to_try.append(os.path.join(base_path, filename))

    # Return the first one that exists
    for path in paths_to_try:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"File '{filename}' not found in bundle or external directory.")

def create_fol(path):
    if not os.path.exists(path): 
        os.makedirs(path)
    else:
        print('already create '+path)

def find_fol(main_input):
    dir_list_input = glob.glob(main_input+'/**/'+'Camera/',recursive=True)
    for i in range(len(dir_list_input)):
        dir_list_input[i] = dir_list_input[i].replace('\\','/')
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

class OverlayGAN():
    def __init__(self, config_data, config_name):
        print('Start read '+config_name+' config...')
        self.coor = config_data["coor"]
        self.mask = config_data["mask"]
        self.img_top_h = config_data["img_top_h"]
        self.img_top_w = config_data["img_top_w"]
        self.img_pano_h =  config_data["img_pano_h"]
        self.img_pano_w = config_data["img_pano_w"]
        self.seamless_center_h = config_data["seamless_center_h"]
        self.seamless_center_w = config_data["seamless_center_w"]
        self.move_img_up = config_data["move_img_up"]
        self.move_img_down = config_data["move_img_down"]
        self.x_new_coor = config_data["x_new_coor"]
        self.y_new_coor = config_data["y_new_coor"]
        self.relate_dist = config_data["relate_dist"]
        self.model_input = config_data["model_input"]
        # for model
        mask_img = cv2.resize(cv2.imread(get_file_path(self.mask)),(self.img_top_h,self.img_top_w)) # for model
        mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
        self.mask_img = mask_img
        mask_img_crop = crop_center(mask_img,(self.model_input[0],self.model_input[1])) # resize to 290,190
        mask_img = preprocess_img(mask_img_crop,img_size=(self.model_input[0],self.model_input[1]))
        mask_bw = cv2.bitwise_not(mask_img) # for only u-net
        self.in_mask = mask_bw.astype('float32')
        # for overlay pano
        mask_img_full = cv2.resize(cv2.imread(get_file_path(config_data["mask_seamless"])),(self.img_top_h,self.img_top_w)) # for model
        mask_img_full = np.where(mask_img_full > 127, 255, 0).astype(np.uint8)
        # seamless mask
        self.mask_img_seamless = cv2.resize(cv2.imread(get_file_path(config_data["mask_seamless"])),(self.img_top_h,self.img_top_w)) # for model
        # project to equirectangular (get mask image)
        equ = Equirectangular(mask_img_full, 167, self.coor, -90,'img')    
        mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        self.mask_pano_img = mask_pano_img.astype(np.uint8)
        # initial generator model
        self.model_type = config_data["model_type"]
        keras.backend.clear_session()
        if(self.model_type == "p2p"):
            self.generator = p2pG.Generator(input_shape=[self.model_input[0],self.model_input[1],3])
            self.generator.load_weights(get_file_path(config_data["model_weight"])) # 500
        elif(self.model_type == "df"):
            self.generator = InpaintGenerator()
            # G.load_weights(args.checkpoint_dir).expect_partial()
            # print(G.history())
            # G.load_weights(args.checkpoint_dir)
            # checkpoint = tf.train.Checkpoint(G)
            # checkpoint.restore(args.checkpoint_dir)
            dummy_input = tf.zeros([1, 512, 512, 3])
            dummy_mask = tf.zeros([1, 512, 512, 1])
            _, _, _ = self.generator(dummy_input,dummy_mask, training=False)
            # for v in G.trainable_variables:
            #     print(v.name, v.shape)
            # with h5py.File('G.weights.h5', 'r') as f:
            #     print(list(f.keys()))
            self.generator.load_weights(get_file_path(config_data["model_weight"]))
            self.df_mask = gen_tranparent_mask(binary_change(mask_img_crop))
            # Convert transparent mask to grayscale
            if self.df_mask.shape[-1] == 4:  # Check if the mask has an alpha channel
                self.df_mask = cv2.cvtColor(self.df_mask, cv2.COLOR_RGBA2GRAY)
            self.df_mask = np.expand_dims(self.df_mask, 2)
            self.df_mask = np.expand_dims(self.df_mask, 0).astype(np.float32)
            self.df_mask = self.df_mask/255.0
        print('Complete setting '+config_name)


    def seamless(self,source,target):
        mask = self.mask_img_seamless
        (h, w) = (self.seamless_center_h, self.seamless_center_w)
        center = (h,w)
        # center = (1002,901)
        im_clone = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
        return im_clone

    # def save_per(self,img_path,save_path,count):
    #     car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
    #     equ = Perspective(car_pano_img) 
    #     per_img_1000 = equ.GetPerspective(167, self.coor, -90, 1500, 1500)
    #     input_image_im = Image.fromarray((per_img_1000).astype(np.uint8))
    #     input_image_im.save(save_path+count+"_topview_source.jpg")
    

    def equi_to_per(self,img_path,name='',save_path=''):
        car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
        equ = Perspective(car_pano_img)
        per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)
        if(name != ''):
            cv2.imwrite(save_path + '/'+ "img_"+str(name).rjust(6, '0')+"_top.jpg", cv2.resize(per_img_1000,(1000,1000)))
        return per_img_1000

    def per_to_equi(self,save_path,car_path,per_img,count):
        car_pano_img = preprocess_img(car_path,img_size=(self.img_pano_w,self.img_pano_h))

        equ = Equirectangular(per_img,167, self.coor, -90)    
        inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
        # save inpaint img (result)
        inpaint_fill = inpaint_pano(inpaint_pano_img,self.mask_pano_img,car_pano_img)
        impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
        impainted_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg")
        # save inpaint img (inpaint mask)
        inpaint_mask = inpaint_pano(self.mask_pano_img,self.mask_pano_img,car_pano_img)
        mask_image_im = Image.fromarray(((inpaint_mask)*255.0).astype(np.uint8))
        mask_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_mask.jpg")
        # inpaint_fill = (inpaint_fill*255.0).astype(np.uint8)
        # cv2.imwrite(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg", inpaint_fill)

    def crude_inpaint(self,source, target):
        mask = self.mask_img
        reverse_mask = cv2.bitwise_not(mask)
        source = source/255.0
        target = target/255.0
        reverse_mask[reverse_mask != 0] = 1
        result = reverse_mask * target + (1 - reverse_mask) * source
        return result
    
    def overlay(self,main_path,save_path,target_coor):
        # try:
        # for i,dir_input in enumerate(dir_list_input):
        all_time = 0.0
        img_path = main_path + "img_"+str(target_coor).rjust(6, '0')+".jpg"
        # print(img_path)
        start_pano = time.time()

        # load image with preprocess
        car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))

        # project to perspective (get car image)
        start_proj = time.time()
        equ = Perspective(car_pano_img)    # Load equirectangular image
        per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)  # Specify parameters(FOV, theta, phi, height, width)
        cv2.imwrite(save_path + '/'+ "img_"+str(target_coor).rjust(6, '0')+"_top.jpg", cv2.resize(cv2.cvtColor(per_img_1000, cv2.COLOR_BGR2RGB),(1000,1000)))
        end_proj = time.time()
        # print('proj: ',end_proj - start_proj)

        # car image
        per_img = crop_center(per_img_1000,(self.model_input[0],self.model_input[1])) # resize to 290,190
        car_img_top = preprocess_img(per_img,img_size=(self.model_input[0],self.model_input[1])) # resize to match input model (256,256)
        cv2.imwrite(save_path + '/'+ "img_"+str(target_coor).rjust(6, '0')+"_crop.jpg", cv2.cvtColor(car_img_top, cv2.COLOR_BGR2RGB))
        car_img_top = car_img_top.astype('float32')


        testgen = createAugment(np.array([car_img_top]), np.array([car_img_top]),np.array([self.in_mask]),batch_size=1,dim=(self.model_input[0],self.model_input[1]), shuffle=False, random_mask=False)
        [masked_image, mask], car_img = testgen[0]
        # plt.imshow(masked_image[0])
        # plt.show()

        if(self.model_type == "p2p"):
            # pix2pix ------------------------------------
            car_img = (car_img - 0.5)/0.5 # normalize [-1,1]
            masked_image = (masked_image - 0.5)/0.5 # normalize [-1,1]
            inputs = masked_image # pix2pix
            start_model = time.time()
            predict_image = self.generator(inputs, training=True)
            end_model = time.time()
            # print('model inpaint time: ',(end_model - start_model))
            predict_image_norm = ((predict_image[0]* 0.5) + 0.5).numpy()# de-normalize [0,1]
            # plt.imshow(predict_image_norm)
            # plt.show()
            masked_image = (masked_image* 0.5) + 0.5 # de-normalize [0,1]
            car_img = (car_img* 0.5) + 0.5 # de-normalize [0,1]
            # ------------------------------------------------
        elif(self.model_type == "df"):
            image = np.expand_dims(car_img_top, 0).astype(np.float32)
            # mask = np.expand_dims(mask, 0).astype(np.float32)
            image = image / 255.0
            # plt.imshow(image[0])
            # plt.show()
            masked_image = image*(1. - self.df_mask)
            cv2.imwrite(save_path + '/'+ "img_"+str(target_coor).rjust(6, '0')+"_mask_top.jpg", cv2.cvtColor(masked_image[0]*255, cv2.COLOR_BGR2RGB))
            plt.imshow(masked_image[0])
            plt.show()
            # mask = mask / 255.0
            x1, x2, offset_flow = self.generator(masked_image, self.df_mask, training=False)
            batch_predicted = x2
            result = batch_predicted  * self.df_mask + masked_image * (1. - self.df_mask)
            predict_image_norm = np.array(batch_predicted[0])
            # plt.imshow(predict_image_norm)
            # plt.show()

        per_img_1000 = per_img_1000/255.0

        start_pro_img = time.time()
        predict_image_norm = preprocess_img(predict_image_norm,img_size=(self.model_input[0],self.model_input[1])) # resize to match input model (256,256)
        predict_image_norm = pad_images_to_same_size(predict_image_norm)
        # predict_image_norm = ((1-self.mask_img) * per_img_1000) + ((self.mask_img) * predict_image_norm)
        # print(predict_image_norm.shape)
        predict_image_norm = self.seamless((predict_image_norm*255.0).astype(np.uint8),(per_img_1000*255.0).astype(np.uint8))
        cv2.imwrite(save_path + '/'+ "img_"+str(target_coor).rjust(6, '0')+"_top_result.jpg", cv2.cvtColor((predict_image_norm).astype(np.uint8), cv2.COLOR_BGR2RGB) )
        # predict_image_norm = predict_image_norm/255.0
        end_pro_img = time.time()
        # print('overlay: ',end_pro_img - start_pro_img)
        self.per_to_equi(save_path, main_path + "img_"+str(target_coor).rjust(6, '0')+".jpg",predict_image_norm,target_coor)

        # start_reproj = time.time()
        # equ = Equirectangular(predict_image_norm,167, self.coor, -90)
        # inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        # end_reproj = time.time()
        # print('reproj: ',(end_reproj - start_reproj)) 

        # inpaint_fill = inpaint_pano(inpaint_pano_img*255.0,self.mask_pano_img,car_pano_img)
        # # inpaint_mask = inpaint_pano(mask_pano_img,mask_pano_img,car_pano_img)
        # end_pano = time.time()

        # # if(output_mode=="save"):
        # save_path = save_path +  "/img_"+str(target_coor).rjust(6, '0')+"_O.jpg"
        # print(save_path)
        # start_save = time.time()
        # self.save_img(path=save_path,inpaint_img=inpaint_fill)
        # end_save = time.time()
        # print('save_time: ',(end_save - start_save))
        # all_time = all_time + (end_pano - start_pano)
        # print('all_time: ',all_time)

    def overlay_process(self,main_path,save_path,overwrite):
        print(main_path)
        target = []
        dir_list_input = glob.glob(main_path+'*'+'.jpg')
        dir_list_input.sort()
        if(not os.path.exists(save_path)):
            create_fol(save_path)
        else:
            if(not bool(strtobool(overwrite))):
                for item in dir_list_input[:]: 
                    pic_full_name = item.split('\\')[-1]
                    pic_name = save_path+'/'+pic_full_name[:-4]+'_O.jpg'
                    if(os.path.exists(pic_name)):
                        dir_list_input.remove(item)
        if(len(dir_list_input)>0):
            for i in range(len(dir_list_input)):
                name_file = (dir_list_input[i].split('\\'))[1]
                num_file = ((name_file.split('_'))[1])[:-4]
                target.append(int(num_file))
        
            for coor in tqdm(target):
                # source_pic = self.find_relate_new(coor_detail,coor,"utm")
                self.overlay(main_path,save_path,coor)

def main():
    # print(_argparse().dir)
    # main_input = "E:/ProcessMMS/" #NOTE: change man path here
    config_path = get_file_path("config_gan_ui.json")
    with open(config_path, "r") as f:
        configs = json.load(f)
    config_name = _argparse().config
    config_data = configs[config_name]
    over_class = OverlayGAN(config_data,config_name)
    main_input = _argparse().dir
    path_input = find_fol(main_input)
    for i in range(len(path_input)):
        line_name = os.path.basename(os.path.dirname(os.path.dirname(path_input[i])))
        save_path = path_input[i]+line_name+'_output'
        # print(save_path)
        
        over_class.overlay_process(path_input[i],save_path,_argparse().overwrite)

if __name__ == '__main__':
    main()