import tkinter as tk
from tkinter import font
from tkinter import filedialog, messagebox, ttk
from tensorflow import keras
import tensorflow as tf
import cv2
import os
import numpy as np
from PIL import Image, ImageTk, ImageDraw
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
import matplotlib.pyplot as plt
import itertools
# import yaml
# tool
from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from tools.process_pano import Equirectangular,Perspective,inpaint_pano
from tools.sam_tools.classification_torch import bbox
from tools.py360convert.py360convert import c2e, e2c
# upscale Real ESRGAN
# from Real_ESRGAN.inference_realesrgan_func import main_upscale
# from initial_gps import *
import os
import sys
import glob
import time
import threading
import json

# import matplotlib.pyplot as plt

event_obj = threading.Event()

# -------------- utility tools --------------------------------------

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

def up_rescale(img,img_size):
    result_img = cv2.resize(img,img_size)
    return(result_img)

def esr_upscale(img,img_size):
    # result_img = main_upscale(img)
    result_img = cv2.resize(img,img_size)
    return(result_img)


# ----------------- initial model setting ---------------
def set_model(model_type,model_input_2d,model_weight):
    print('start model')
    # initial generator model
    keras.backend.clear_session()
    if(model_type == "p2p"):
        generator = p2pG.Generator(input_shape=[model_input_2d[0],model_input_2d[1],3])
        generator.load_weights(get_file_path(model_weight)) # 500
    elif(model_type == "df"):
        generator = InpaintGenerator()
        # G.load_weights(args.checkpoint_dir).expect_partial()
        # print(G.history())
        # G.load_weights(args.checkpoint_dir)
        # checkpoint = tf.train.Checkpoint(G)
        # checkpoint.restore(args.checkpoint_dir)
        dummy_input = tf.zeros([1, 512, 512, 3])
        dummy_mask = tf.zeros([1, 512, 512, 1])
        _, _, _ = generator(dummy_input,dummy_mask, training=False)
        # for v in G.trainable_variables:
        #     print(v.name, v.shape)
        # with h5py.File('G.weights.h5', 'r') as f:
        #     print(list(f.keys()))
        generator.load_weights(get_file_path(model_weight))
    print('set_model: ',model_type)
    print('set_model: ',generator)
    print('model start success')
    return generator

# ------------------- overlay function -------------------------

class OverlayGAN():
    def __init__(self, config_data, config_name):
        print('Start read '+config_name+' config...')
        self.config_data = config_data
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
        self.model_type = config_data["model_type"]
        self.model_weight = config_data["model_weight"]

        # ------- mask value -----------------------
        self.mask_img = None
        self.in_mask = None
        self.mask_img_seamless = None
        self.mask_pano_img = None
        self.df_mask = None
        # ------------------------------------------

        # self.model_type = None
        self.generator = None
        self.set_useMask()
        
        # # initial generator model
        # self.model_type = config_data["model_type"]
        # keras.backend.clear_session()
        # if(self.model_type == "p2p"):
        #     self.generator = p2pG.Generator(input_shape=[self.model_input[0],self.model_input[1],3])
        #     self.generator.load_weights(get_file_path(config_data["model_weight"])) # 500
        # elif(self.model_type == "df"):
        #     self.generator = InpaintGenerator()
        #     # G.load_weights(args.checkpoint_dir).expect_partial()
        #     # print(G.history())
        #     # G.load_weights(args.checkpoint_dir)
        #     # checkpoint = tf.train.Checkpoint(G)
        #     # checkpoint.restore(args.checkpoint_dir)
        #     dummy_input = tf.zeros([1, 512, 512, 3])
        #     dummy_mask = tf.zeros([1, 512, 512, 1])
        #     _, _, _ = self.generator(dummy_input,dummy_mask, training=False)
        #     # for v in G.trainable_variables:
        #     #     print(v.name, v.shape)
        #     # with h5py.File('G.weights.h5', 'r') as f:
        #     #     print(list(f.keys()))
        #     self.generator.load_weights(get_file_path(config_data["model_weight"]))
        print('Complete reading config '+config_name)

    # def get_model(self):
    #     print('get_model: ',self.model_type)
    #     print('get_model: ',self.generator)
    #     return (self.generator)

    def set_model(self,generator):
        self.generator = generator
        print('superclass: ',self.generator)
        

    def set_segMask(self,seg_mask):
        mask = np.array(seg_mask) # grayscale to ensure image is binary
        mask_ori = binary_change(mask[:])
        mask_bw = cv2.bitwise_not(mask_ori[:])
        self.in_mask = mask_ori
        print('mask original')
        # plt.imshow(mask_ori)
        # plt.show()
        print('mask bitwise')
        # pad_mask = pad_images_to_same_size(mask_ori/255.0) # pad mask to match model input size
        pad_mask = up_rescale(mask_ori/255.0,(self.img_top_w,self.img_top_h)) # resize mask to match model input size
        self.mask_img = pad_mask
        self.mask_img_seamless = (pad_mask[:]*255.0).astype(np.uint8)
        # plt.imshow(cv2.resize(np.array(pad_mask),(512,512)))
        # plt.show()
        print('pad mask')
        # ----------- change to panorama mask ---------------
        #plt.imshow(cv2.resize(np.array(pad_mask),(512,512)))
        #plt.show()
        equ = Equirectangular(pad_mask[:]*255.0, 167, self.coor, -90,'img')    
        mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        self.mask_pano_img = mask_pano_img.astype(np.uint8)
        #plt.imshow(mask_bw)
        #plt.show()
        # --------------- transparent mask for model ---------------
        self.df_mask = gen_tranparent_mask(mask_ori)
        # Convert transparent mask to grayscale
        if self.df_mask.shape[-1] == 4:  # Check if the mask has an alpha channel
            self.df_mask = cv2.cvtColor(self.df_mask, cv2.COLOR_RGBA2GRAY)
        self.df_mask = np.expand_dims(self.df_mask, 2)
        self.df_mask = np.expand_dims(self.df_mask, 0).astype(np.float32)
        self.df_mask = self.df_mask/255.0

    def set_useMask(self):
        # for model
        mask_img = cv2.resize(cv2.imread(get_file_path(self.mask)),(self.img_top_h,self.img_top_w)) # for model
        mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
        self.mask_img = mask_img # read from full top mask img
        mask_img_crop = crop_center(mask_img,(self.model_input[0],self.model_input[1])) # resize to 290,190
        mask_img = preprocess_img(mask_img_crop,img_size=(self.model_input[0],self.model_input[1]))
        mask_bw = cv2.bitwise_not(mask_img) # for only u-net
        self.in_mask = mask_bw.astype('float32') # crop to model input
        # for overlay pano
        mask_img_full = cv2.resize(cv2.imread(get_file_path(self.config_data["mask_seamless"])),(self.img_top_h,self.img_top_w)) # for model
        mask_img_full = np.where(mask_img_full > 127, 255, 0).astype(np.uint8)
        # seamless mask
        self.mask_img_seamless = cv2.resize(cv2.imread(get_file_path(self.config_data["mask_seamless"])),(self.img_top_h,self.img_top_w)) # for model
        # project to equirectangular (get mask image)
        equ = Equirectangular(mask_img_full, 167, self.coor, -90,'img')    
        mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        self.mask_pano_img = mask_pano_img.astype(np.uint8)

        self.df_mask = gen_tranparent_mask(binary_change(mask_img_crop))
        # Convert transparent mask to grayscale
        if self.df_mask.shape[-1] == 4:  # Check if the mask has an alpha channel
            self.df_mask = cv2.cvtColor(self.df_mask, cv2.COLOR_RGBA2GRAY)
        self.df_mask = np.expand_dims(self.df_mask, 2)
        self.df_mask = np.expand_dims(self.df_mask, 0).astype(np.float32)
        self.df_mask = self.df_mask/255.0


    def seamless(self,source,target,mask):
        print('seamless function process')
        # mask = self.mask_img_seamless
        print(type(source),' ',type(target),' ',type(mask))
        print(source.dtype,' ',target.dtype,' ',mask.dtype)
        print(source.shape,' ',target.shape,' ',mask.shape)
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

        # Compute center of the bounding box
        center = (x + w // 2, y + h // 2)
        # (h, w) = (self.seamless_center_h, self.seamless_center_w)
        # center = (h,w)
        # center = (1002,901)
        # plt.imshow(np.array(source))
        # plt.show()
        # a = Image.fromarray(source.astype(np.uint8))
        # a.save("D:/KMUTNB/KMUTNB/test_pic/old/Camera/old_output" + '/'+ "img_"+str(0).rjust(6, '0')+"_source.jpg")
        # plt.imshow(np.array(target))
        # plt.show()
        # b = Image.fromarray(((target)).astype(np.uint8))
        # b.save("D:/KMUTNB/KMUTNB/test_pic/old/Camera/old_output" + '/'+ "img_"+str(0).rjust(6, '0')+"_target.jpg")
        # plt.imshow(np.array(mask))
        # plt.show()
        # c = Image.fromarray(((mask)).astype(np.uint8))
        # c.save("D:/KMUTNB/KMUTNB/test_pic/old/Camera/old_output" + '/'+ "img_"+str(0).rjust(6, '0')+"_mask_seam.jpg")
        im_clone = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
        # d = Image.fromarray(((im_clone)).astype(np.uint8))
        # d.save("D:/KMUTNB/KMUTNB/test_pic/old/Camera/old_output" + '/'+ "img_"+str(0).rjust(6, '0')+"_out_seam.jpg")
        # plt.imshow(np.array(im_clone))
        # plt.show()
        return im_clone

    # def save_per(self,img_path,save_path,count):
    #     car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
    #     equ = Perspective(car_pano_img) 
    #     per_img_1000 = equ.GetPerspective(167, self.coor, -90, 1500, 1500)
    #     input_image_im = Image.fromarray((per_img_1000).astype(np.uint8))
    #     input_image_im.save(save_path+count+"_topview_source.jpg")
    

    def equi_to_per(self,img_path,name=''):
        car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
        equ = Perspective(car_pano_img)
        per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)
        if(name != ''):
            cv2.imwrite("D:/new_car_ds/pano_8k/image_top/"+name+".jpg", per_img_1000)
        return per_img_1000

    def per_to_equi(self,save_path,car_path,per_img,count):
        car_pano_img = preprocess_img(car_path,img_size=(self.img_pano_w,self.img_pano_h))

        equ = Equirectangular(per_img,167, self.coor, -90)    
        inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
        inpaint_fill = inpaint_pano(inpaint_pano_img,self.mask_pano_img,car_pano_img)
        impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
        impainted_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg")
        #save inpaint img (inpaint mask)
        inpaint_mask = inpaint_pano(self.mask_pano_img,self.mask_pano_img,car_pano_img)
        mask_image_im = Image.fromarray(((inpaint_mask)*255.0).astype(np.uint8))
        mask_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_mask_"+'top'+".jpg")
        # save binary mask panorama
        bmask_image_im = Image.fromarray(((self.mask_pano_img)).astype(np.uint8))
        bmask_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_bmask_"+'top'+".jpg")
        # inpaint_fill = (inpaint_fill*255.0).astype(np.uint8)
        # cv2.imwrite(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg", inpaint_fill)
        return inpaint_fill

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
        end_proj = time.time()
        # print('proj: ',end_proj - start_proj)

        # car image
        # per_img = crop_center(per_img_1000,(self.model_input[0],self.model_input[1])) # crop image to model input size
        per_img = up_rescale(per_img_1000,(self.model_input[0],self.model_input[1])) # resize image to match model input size
        car_img_top = preprocess_img(per_img,img_size=(self.model_input[0],self.model_input[1])) # resize to match input model (256,256)

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
            # plt.imshow(masked_image[0])
            # plt.show()
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
        # predict_image_norm = pad_images_to_same_size(predict_image_norm) # pad image to match model input size
        predict_image_norm = esr_upscale(predict_image_norm,(self.img_top_w, self.img_top_h)) # resize image to match real input size
        # predict_image_norm = ((1 - mask_img_pano) * per_img_1000) + (mask_img_pano * predict_image_norm)
        predict_image_norm = self.seamless((predict_image_norm*255.0).astype(np.uint8),(per_img_1000*255.0).astype(np.uint8),self.mask_img_seamless)
        # predict_image_norm = predict_image_norm/255.0
        end_pro_img = time.time()
        # print('overlay: ',end_pro_img - start_pro_img)
        final_pano_output = self.per_to_equi(save_path, main_path + "img_"+str(target_coor).rjust(6, '0')+".jpg",predict_image_norm,target_coor)
        return final_pano_output

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

    # def overlay_process(self,main_path,save_path, coor_detail,overwrite):
    #     print(main_path)
    #     target = []
    #     dir_list_input = glob.glob(main_path+'*'+'.jpg')
    #     dir_list_input.sort()
    #     if(not os.path.exists(save_path)):
    #         create_fol(save_path)
    #     else:
    #         if(not bool(strtobool(overwrite))):
    #             for item in dir_list_input[:]: 
    #                 pic_full_name = item.split('/')[-1]
    #                 pic_name = save_path+'/'+pic_full_name[:-4]+'_O.jpg'
    #                 if(os.path.exists(pic_name)):
    #                     dir_list_input.remove(item)
    #     if(len(dir_list_input)>0):
    #         for i in range(len(dir_list_input)):
    #             name_file = (dir_list_input[i].split('/'))[1]
    #             num_file = ((name_file.split('_'))[1])[:-4]
    #             target.append(int(num_file))
        
    #         for coor in tqdm(target):
    #             source_pic = self.find_relate_new(coor_detail,coor,"utm")
    #             self.overlay(main_path,save_path,source_pic,coor,str(coor),coor_detail)

# --------------- inherited for cube overlay -------------------------
class OverlayCube(OverlayGAN):
    def __init__(self,config_data, config_name, generator=None):
        super().__init__(config_data, config_name)
        self.generator = generator
        self.in_mask = {
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.mask_img = {
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.mask_img_seamless = {
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.df_mask = {
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.have_mask = {
                "cube_front": False,
                "cube_right": False,
                "cube_back": False,
                "cube_left": False,
                "cube_top": False,
                "cube_bottom": False
            }
        

    # def check_data(self):
    #     print(self.config_data)
    #     print(self.coor)
    #     print('cube class: ',self.generator)
    def set_segMask(self,seg_mask):
        for count, side_name in enumerate(seg_mask):
            if(seg_mask[side_name] is not None):
                mask = np.array(seg_mask[side_name]) # grayscale to ensure image is binary
                self.have_mask[side_name] = True
            else:
                mask = np.full((self.img_top_h,self.img_top_w,3), 0.0, np.float32)
            mask_ori = binary_change(mask[:])
            mask_bw = cv2.bitwise_not(mask_ori[:])
            self.in_mask[side_name] = mask_ori
            print('mask original')
            # plt.imshow(mask_ori)
            # plt.show()
            print('mask bitwise')
            # pad_mask = pad_images_to_same_size(mask_ori/255.0) # pad mask to match model input size
            pad_mask = up_rescale(mask_ori/255.0,(self.img_top_w,self.img_top_h)) # resize mask to match model input size
            self.mask_img[side_name] = pad_mask
            self.mask_img_seamless[side_name] = (pad_mask[:]*255.0).astype(np.uint8)
            # plt.imshow(cv2.resize(np.array(pad_mask),(512,512)))
            # plt.show()
            print('pad mask')
        # ----------- change to panorama mask ---------------
        #plt.imshow(cv2.resize(np.array(pad_mask),(512,512)))
        #plt.show()
        # equ = Equirectangular(pad_mask[:]*255.0, 167, self.coor, -90,'img')    
        # mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        print(len(list(self.mask_img.values())))
        # check_cube_list = list(self.mask_img.values())
        # for i in range(len(check_cube_list)):
        #     plt.imshow(cv2.resize(check_cube_list[i], (512,512)))
        self.mask_pano_img = c2e(list(self.mask_img.values()), h=self.img_pano_h, w=self.img_pano_w, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        self.mask_pano_img = self.mask_pano_img*255.0
        # plt.imshow(cv2.resize(self.mask_pano_img,(256,128)))
        # plt.show()
        # --------------- transparent mask for model ---------------
        for count, side_name in enumerate(seg_mask):
            self.df_mask[side_name] = gen_tranparent_mask(self.in_mask[side_name])
            # Convert transparent mask to grayscale
            if self.df_mask[side_name].shape[-1] == 4:  # Check if the mask has an alpha channel
                self.df_mask[side_name] = cv2.cvtColor(self.df_mask[side_name], cv2.COLOR_RGBA2GRAY)
            self.df_mask[side_name] = np.expand_dims(self.df_mask[side_name], 2)
            self.df_mask[side_name] = np.expand_dims(self.df_mask[side_name], 0).astype(np.float32)
            self.df_mask[side_name] = self.df_mask[side_name]/255.0

    # def set_useMask(self):
    #     # for model
    #     mask_img = cv2.resize(cv2.imread(get_file_path(self.mask)),(self.img_top_h,self.img_top_w)) # for model
    #     mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
    #     self.mask_img = mask_img # read from full top mask img
    #     mask_img_crop = crop_center(mask_img,(self.model_input[0],self.model_input[1])) # resize to 290,190
    #     mask_img = preprocess_img(mask_img_crop,img_size=(self.model_input[0],self.model_input[1]))
    #     mask_bw = cv2.bitwise_not(mask_img) # for only u-net
    #     self.in_mask = mask_bw.astype('float32') # crop to model input
    #     # for overlay pano
    #     mask_img_full = cv2.resize(cv2.imread(get_file_path(self.config_data["mask_seamless"])),(self.img_top_h,self.img_top_w)) # for model
    #     mask_img_full = np.where(mask_img_full > 127, 255, 0).astype(np.uint8)
    #     # seamless mask
    #     self.mask_img_seamless = cv2.resize(cv2.imread(get_file_path(self.config_data["mask_seamless"])),(self.img_top_h,self.img_top_w)) # for model
    #     # project to equirectangular (get mask image)
    #     equ = Equirectangular(mask_img_full, 167, self.coor, -90,'img')    
    #     mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
    #     self.mask_pano_img = mask_pano_img.astype(np.uint8)

    #     self.df_mask = gen_tranparent_mask(binary_change(mask_img_crop))
    #     # Convert transparent mask to grayscale
    #     if self.df_mask.shape[-1] == 4:  # Check if the mask has an alpha channel
    #         self.df_mask = cv2.cvtColor(self.df_mask, cv2.COLOR_RGBA2GRAY)
    #     self.df_mask = np.expand_dims(self.df_mask, 2)
    #     self.df_mask = np.expand_dims(self.df_mask, 0).astype(np.float32)
    #     self.df_mask = self.df_mask/255.0

    def per_to_equi(self,save_path,car_path,per_img,count):
        if isinstance(car_path, str):
            car_pano_img = preprocess_img(car_path,img_size=(self.img_pano_w,self.img_pano_h))
        else:
            car_pano_img = car_path
        inpaint_pano_img = per_img.astype(np.uint8)
        # inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
        # equ = Equirectangular(per_img,167, self.coor, -90)    
        # inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        # inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
        inpaint_fill = inpaint_pano(inpaint_pano_img,self.mask_pano_img,car_pano_img)
        impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
        impainted_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg")
        #save inpaint img (inpaint mask)
        inpaint_mask = inpaint_pano(self.mask_pano_img,self.mask_pano_img,car_pano_img)
        mask_image_im = Image.fromarray(((inpaint_mask)*255.0).astype(np.uint8))
        mask_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_mask_"+'top'+".jpg")
        # save binary mask panorama
        bmask_image_im = Image.fromarray(((self.mask_pano_img)).astype(np.uint8))
        bmask_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_bmask_"+'top'+".jpg")
        # inpaint_fill = (inpaint_fill*255.0).astype(np.uint8)
        # cv2.imwrite(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg", inpaint_fill)
        return inpaint_fill

    def overlay(self,pic_path,save_path,target_coor):
        output_cube = []
        # try:
        # for i,dir_input in enumerate(dir_list_input):
        all_time = 0.0
        if isinstance(pic_path, str):
            car_pano_img = pic_path + "img_"+str(target_coor).rjust(6, '0')+".jpg"
            # load image with preprocess
            car_pano_img = preprocess_img(car_pano_img,img_size=(self.img_pano_w,self.img_pano_h))
        else:
            car_pano_img = (np.array(pic_path*255)).astype(np.uint8)
        # print(img_path)
        start_pano = time.time()

        # project to perspective (get car image)
        start_proj = time.time()
        # equ = Perspective(car_pano_img)    # Load equirectangular image
        # per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)  # Specify parameters(FOV, theta, phi, height, width)
        image_cube = e2c(car_pano_img, face_w=1000, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        end_proj = time.time()
        # print('proj: ',end_proj - start_proj)

        # car image
        for count, side_name in enumerate(self.have_mask):
            # plt.imshow(image_cube[count])
            # plt.show()
            if(self.have_mask[side_name]):
                # per_img = crop_center(image_cube[count],(self.model_input[0],self.model_input[1])) # pad image to match model input size
                per_img = up_rescale(image_cube[count],(self.model_input[0],self.model_input[1])) # resize image to match model input size
                car_img_top = preprocess_img(per_img,img_size=(self.model_input[0],self.model_input[1])) # resize to match input model (256,256)

                car_img_top = car_img_top.astype('float32')

                print(side_name,': ',np.array([car_img_top]).shape,' ',np.array([self.in_mask[side_name]]).shape)
                testgen = createAugment(np.array([car_img_top]), np.array([car_img_top]),np.array([self.in_mask[side_name]]),batch_size=1,dim=(self.model_input[0],self.model_input[1]), shuffle=False, random_mask=False)
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
                    masked_image = image*(1. - self.df_mask[side_name])
                    # plt.imshow(masked_image[0])
                    # plt.show()
                    # mask = mask / 255.0
                    x1, x2, offset_flow = self.generator(masked_image, self.df_mask[side_name], training=False)
                    batch_predicted = x2
                    result = batch_predicted  * self.df_mask[side_name] + masked_image * (1. - self.df_mask[side_name])
                    predict_image_norm = np.array(batch_predicted[0])
                    # print('--------------------',side_name)
                    # plt.imshow(predict_image_norm)
                    # plt.show()

                # per_img_1000 = per_img_1000/255.0

                start_pro_img = time.time()
                predict_image_norm = preprocess_img(predict_image_norm,img_size=(self.model_input[0],self.model_input[1])) # resize to match input model (256,256)
                # predict_image_norm = pad_images_to_same_size(predict_image_norm) # pad mask to match model input size
                predict_image_norm = esr_upscale(predict_image_norm,(self.img_top_w, self.img_top_h)) # resize mask to match model input size
                # predict_image_norm = ((1 - mask_img_pano) * per_img_1000) + (mask_img_pano * predict_image_norm)
                predict_image_norm = self.seamless((predict_image_norm*255.0).astype(np.uint8),(image_cube[count]).astype(np.uint8),self.mask_img_seamless[side_name])
                # predict_image_norm = predict_image_norm/255.0
                end_pro_img = time.time()
                # print('overlay: ',end_pro_img - start_pro_img)
                output_cube.append(predict_image_norm)
            else:
                output_cube.append(image_cube[count])
        cube_pano_img = c2e(output_cube, h=self.img_pano_h, w=self.img_pano_w, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom] # pane inpaint cube to panorama
        # plt.imshow(cv2.resize(car_pano_img,(256,128)))
        # plt.show()
        # plt.imshow(cv2.resize(cube_pano_img,(256,128)))
        # plt.show()
        # plt.imshow(cv2.resize(target_coor,(256,128)))
        # plt.show()
        final_pano_output = self.per_to_equi(save_path, car_pano_img,cube_pano_img,target_coor)

        return final_pano_output

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
# ------------ main windows -------------------------------------

class MainApp(tk.Tk):
    def __init__(self,configs):
        super().__init__()
        self.title("Inpaint System Ver.1.4")
        self.geometry("700x550")
        icon_path = "icon.ico"
        self.iconbitmap(get_file_path(icon_path))

        # container holds all pages
        self.container = tk.Frame(self)
        self.container.pack()

        print(configs.get(list(configs.keys())[0], {}))

        # share data
        self.shared_data = {
            "input_img": {
                "top_view": None,
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            },
            "segment_mask" : {
                "top_view": None,
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            },
            "config_data": configs.get(list(configs.keys())[0], {})
        }

        # store pages in a dict
        self.frames = {}

        for F in (ImageProcessingApp, PageTwo):
            if(F == ImageProcessingApp):
                frame = F(parent=self.container, controller = self, configs = configs)
            elif(F == PageTwo):
                frame = F(parent=self.container, controller = self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(ImageProcessingApp)

    def show_frame(self, page_class):
        frame = self.frames[page_class]
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()


        # Dynamically resize the window
        if page_class == ImageProcessingApp:
            self.geometry("615x560")
        elif page_class == PageTwo:
            self.geometry("1400x650")

        # Call on_show() if it exists
        if hasattr(frame, "on_show"):
            frame.on_show()

# --------------------- segment page ------------------------

class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="red")
        self.page_ctrl = controller
        # ------------ variable for segment task
        self.config_data = self.page_ctrl.shared_data["config_data"]
        self.image = {
                "top_view": None,
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.tk_image = {
                "top_view": None,
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.image_pano = None
        self.image_pano_cube = None
        self.type_load = None
        self.mask_canvas = None
        self.mask_canvas_draw = None
        self.binary_mask = {
                "top_view": None,
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.start_x,self.start_y,self.end_x,self.end_y = None, None, None, None
        # if(len(self.config_data)>0):
        # ----------- create element in frame --------------------------
        # tk.Label(self, text="Input", font=("Arial", 18)).pack(pady=20)
        self.header_frame = tk.Frame(self, width=700, bg="green")
        self.header_frame.pack(fill="both", padx=10, pady=10)
        self.side_label_show = tk.StringVar()
        self.nav_image = None
        self.btn_left = None
        self.btn_right = None
        # tk.Label(self, textvariable=controller.shared_data["input_text"]).pack()
        # tk.Button(self, text="Go to Page One", command=lambda: controller.show_frame(ImageProcessingApp)).pack()
        # Create main 3-part horizontal layout
        self.left_frame = tk.Frame(self, width=300, bg="white")
        self.center_frame = tk.Frame(self, width=100, bg="lightgray")
        self.right_frame = tk.Frame(self, width=300, bg="white")

        self.left_frame.pack(side="left", fill="both", expand=False)
        self.center_frame.pack(side="left", fill="y")
        self.right_frame.pack(side="left", fill="both", expand=False)

        self.input_canvas = tk.Canvas(self.left_frame)
        self.input_canvas_image = None

        self.seg_canvas = tk.Canvas(self.right_frame)
        self.seg_canvas_image = None

        self.build_left()
        self.build_center()
        self.build_right()
        self.build_header()

        self.input_canvas.bind("<ButtonPress-1>", self.draw_rec_press)
        self.input_canvas.bind("<B1-Motion>", self.draw_rec_move)
        self.input_canvas.bind("<ButtonRelease-1>", self.draw_rec_release)
        # if(self.page_ctrl.shared_data['input_img'] != None):
        #     self.load_image()

    def on_show(self):
        self.config_data = self.page_ctrl.shared_data["config_data"]
        self.pre_process_img()
        self.set_canvas(list(self.image.keys())[0])

    def pre_process_img(self, type_load="top"):
        print(self.config_data)
        # file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        # if not file_path:
        #     return
        
        # ----------------- same as input ---------------------------------
        # self.image = (Image.open(file_path).convert("RGB").resize((512,512))) # open file by pillow
        ## --------------------------------

        # ----------- panorama -> top-view -----------------------------
        self.image_pano = preprocess_img(self.page_ctrl.shared_data['input_img'],img_size=(self.config_data['img_pano_w'],self.config_data['img_pano_h']))
        # if(type_load == 'top'):
        # self.type_load = 'top'
        ## select top view ---------------##
        equ = Perspective(self.image_pano)    # Load equirectangular image
        self.full_top_img = equ.GetPerspective(167, self.config_data['coor'], -90, self.config_data['img_top_h'], self.config_data['img_top_w'])  # Specify parameters(FOV, theta, phi, height, width)
        # car image
        # per_img = crop_center(self.full_top_img,(self.config_data['model_input'][0],self.config_data['model_input'][1])) # crop to match model input size
        per_img = up_rescale(self.full_top_img,(self.config_data['model_input'][0],self.config_data['model_input'][1])) # resize to match model input size
        # Convert to PIL Image
        pil_image = Image.fromarray(per_img.astype('uint8'))
        self.image["top_view"] = pil_image
        self.tk_image["top_view"] = ImageTk.PhotoImage(self.image["top_view"])
        ## --------------------------------
        # elif(type_load == 'cube'):
        #     self.type_load = 'cube'
        # ------------------- panorama -> cube-map -------------------------------
        self.image_pano_cube = e2c(self.image_pano, face_w=1000, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        # car_cube_img = self.image_pano_cube[2]
        # car image
        for count, side_name in enumerate(self.image):
            if(count>0):
                count = count - 1
                # car_cube_img = crop_center(self.image_pano_cube[count],(self.config_data['model_input'][0],self.config_data['model_input'][1])) # crop to match model input size
                car_cube_img = up_rescale(self.image_pano_cube[count],(self.config_data['model_input'][0],self.config_data['model_input'][1])) # resize to match model input size
                self.image[side_name] = Image.fromarray(car_cube_img.astype('uint8'))
                self.tk_image[side_name] = ImageTk.PhotoImage(self.image[side_name])
        self.nav_image = DictNavigator(self.image)
        # car_cube_img = crop_center(self.full_top_img,(self.model_input[0],self.model_input[1])) # resize to 290,190
        # self.image = Image.fromarray(car_cube_img.astype('uint8'))
        # self.tk_image = ImageTk.PhotoImage(self.image)
        # if self.canvas_image_cube:
        #     self.canvas_cube.delete(self.canvas_cube)
        # self.canvas_image_cube = self.canvas_cube.create_image(0, 0, anchor="nw", image=self.tk_image_cube)
        #     # ----------------------------------------

        # self.input_canvas.config(width=self.image.width, height=self.image.height)
        # # self.seg_canvas.config(width=self.image.width, height=self.image.height)

        # if self.input_canvas_image:
        #     self.input_canvas.delete(self.input_canvas_image)
        # self.input_canvas_image = self.input_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        # # self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # self.mask_canvas = Image.new("1", self.image.size, 0)
        # self.mask_canvas_draw = ImageDraw.Draw(self.mask_canvas)

        # self.save_button.config(state=tk.NORMAL)
        # self.inpaint_button.config(state=tk.NORMAL)

    def set_canvas(self,side_select):
        self.input_canvas.config(width=self.image[side_select].width, height=self.image[side_select].height)
        # self.seg_canvas.config(width=self.image.width, height=self.image.height)

        if self.input_canvas_image:
            self.input_canvas.delete(self.input_canvas_image)
        self.input_canvas_image = self.input_canvas.create_image(0, 0, anchor="nw", image=self.tk_image[side_select])
        # self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.mask_canvas = Image.new("1", self.image[side_select].size, 0)
        self.mask_canvas_draw = ImageDraw.Draw(self.mask_canvas)

        self.side_label_show.set(side_select)

        if(self.binary_mask[side_select] is not None):
            overlar_image = ((self.binary_mask[side_select]) * np.array(self.image[side_select])) + np.array(self.image[side_select])
            self.seg_pil_image = Image.fromarray(overlar_image.astype('uint8'))
            self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
        else:
            # self.seg_pil_image = Image.fromarray(self.image[side_select].astype('uint8'))
            self.seg_tk_img = ImageTk.PhotoImage(self.image[side_select])
        self.seg_canvas.config(width=self.image[side_select].width, height=self.image[side_select].height)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)

    def load_mask(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        # if not file_path:
        #     return
        self.binary_mask[self.side_label_show.get()] = preprocess_img(file_path,(self.config_data['img_top_h'], self.config_data['img_top_w'])) # open file by pillow
        self.binary_mask[self.side_label_show.get()] = binary_change(self.binary_mask[self.side_label_show.get()])
        # self.binary_mask[self.side_label_show.get()] = crop_center(self.binary_mask[self.side_label_show.get()],(self.config_data['model_input'][0],self.config_data['model_input'][1])) # crop image to match model input size
        self.binary_mask[self.side_label_show.get()] = up_rescale(self.binary_mask[self.side_label_show.get()],(self.config_data['model_input'][0],self.config_data['model_input'][1]))
        overlar_image = ((self.binary_mask[self.side_label_show.get()]) * np.array(self.image[self.side_label_show.get()])) + np.array(self.image[self.side_label_show.get()])
        # plt.imshow(overlar_image)
        # plt.show()
        self.seg_pil_image = Image.fromarray(overlar_image.astype('uint8'))
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
        self.seg_canvas.config(width=self.image[self.side_label_show.get()].width, height=self.image[self.side_label_show.get()].height)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)

    def paint(self, event):
        if not self.mask_canvas_draw:
            return

        x, y = event.x, event.y
        self.input_canvas.create_oval(x - self.brush_size, y - self.brush_size,
                                x + self.brush_size, y + self.brush_size,
                                fill="red", outline="red")
        self.mask_canvas_draw.ellipse([x - self.brush_size, y - self.brush_size,
                                x + self.brush_size, y + self.brush_size], fill=1)
        
    def draw_rec_press(self,event):
        # if not self.mask_draw:
        #     return
        self.start_x, self.start_y = event.x, event.y
        self.rect = self.input_canvas.create_rectangle(self.start_x,self.start_y,self.start_x,self.start_y,
                                outline="red")
        
    def draw_rec_move(self,event):
        if self.rect:
            self.input_canvas.coords(self.rect,self.start_x,self.start_y,event.x,event.y)
        
    def draw_rec_release(self,event):
        self.end_x = event.x
        self.end_y = event.y
        print(f"draw rectangle xyxy format {self.start_x}, {self.start_y}, {event.x}, {event.y}")
        self.mask_canvas_draw.rectangle([self.start_x, self.start_y,
                                self.end_x, self.end_y], fill=0)
        
    def direct_draw(self):
        mask = np.full((self.image[self.side_label_show.get()].width,self.image[self.side_label_show.get()].height,3), 0, np.uint8) ## black background
        color = (255, 255, 255)
        thickness = -1
        self.binary_mask[self.side_label_show.get()] = cv2.rectangle(mask, (self.start_x, self.start_y), (self.end_x, self.end_y), color, thickness)
        overlar_image = ((self.binary_mask[self.side_label_show.get()]) * np.array(self.image[self.side_label_show.get()])) + np.array(self.image[self.side_label_show.get()])
        # plt.imshow(overlar_image)
        # plt.show()
        self.seg_pil_image = Image.fromarray(overlar_image.astype('uint8'))
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
        self.seg_canvas.config(width=self.image[self.side_label_show.get()].width, height=self.image[self.side_label_show.get()].height)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)

    def get_binary_mask(self):
        if self.mask:
            self.mask = (self.mask).convert('RGB')
            return np.array(self.mask, dtype=np.uint8)
        return None

    def save_mask(self):
        if self.binary_mask[self.side_label_show.get()] is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if save_path:
                # tmp_save = pad_images_to_same_size(self.binary_mask[self.side_label_show.get()]) # pad image to match original size
                tmp_save = up_rescale(self.binary_mask[self.side_label_show.get()],(self.config_data['img_top_w'], self.config_data['img_top_h'])) # resize image to match original image
                tmp_save = Image.fromarray(tmp_save.astype('uint8'))
                tmp_save.save(save_path)
                print("Mask saved to:", save_path)

    def run_segment(self):
        self.binary_mask[self.side_label_show.get()] = bbox("per",np.array(self.image[self.side_label_show.get()]), [self.start_x,self.start_y,self.end_x,self.end_y])
        self.binary_mask[self.side_label_show.get()] = (self.binary_mask[self.side_label_show.get()].squeeze()).astype(np.uint8)
        self.binary_mask[self.side_label_show.get()] = cv2.cvtColor(self.binary_mask[self.side_label_show.get()], cv2.COLOR_GRAY2BGR)
        self.binary_mask[self.side_label_show.get()] = self.resize_mask(self.binary_mask[self.side_label_show.get()])
        # tmp_image = self.image[:]
        overlar_image = ((self.binary_mask[self.side_label_show.get()]) * np.array(self.image[self.side_label_show.get()])) + np.array(self.image[self.side_label_show.get()])
        # plt.imshow(overlar_image)
        # plt.show()
        self.seg_pil_image = Image.fromarray(overlar_image.astype('uint8'))
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
        self.seg_canvas.config(width=self.image[self.side_label_show.get()].width, height=self.image[self.side_label_show.get()].height)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)
        # pil_image = Image.fromarray(per_img.astype('uint8'))
        # self.image = pil_image
        # ## --------------------------------

        # self.tk_image = ImageTk.PhotoImage(self.image)

    def resize_mask(self,mask, scale=1.2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) 
        dilated_mask = cv2.dilate(mask, kernel)
        # h, w = mask.shape[:2]
        # new_size = (int(w * scale), int(h * scale))
        # resized = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        # resized = crop_center(resized,(self.config_data['model_input'][0],self.config_data['model_input'][1]))
        return dilated_mask

    def next_step(self):
        # self.page_ctrl.shared_data["input_text"].set("Hello from segment page")
        self.page_ctrl.shared_data["segment_mask"] = self.binary_mask
        self.page_ctrl.show_frame(ImageProcessingApp)
        # self.update_idletasks()

    def next_image(self):
        key, _ = self.nav_image.next()
        current = self.nav_image.index
        if(current >= len(list(self.image.keys()))):
            self.btn_right.config(state=tk.DISABLED)
            self.btn_left.config(state=tk.NORMAL)
        else:
            self.btn_right.config(state=tk.NORMAL)
            self.btn_left.config(state=tk.NORMAL)
        self.set_canvas(key)
    #     cycled_items = itertools.cycle(self.image)
    #     for key, value in cycled_items:
    #         self.set_canvas(key)

    def previous_image(self):
        key, _ = self.nav_image.previous()
        current = self.nav_image.index
        if(current == 0):
            self.btn_right.config(state=tk.NORMAL)
            self.btn_left.config(state=tk.DISABLED)
        else:
            self.btn_right.config(state=tk.NORMAL)
            self.btn_left.config(state=tk.NORMAL)
        self.set_canvas(key)


    def build_left(self):
        # print(self.page_ctrl.shared_data['input_img'])
        # if(self.page_ctrl.shared_data['input_img'] != None):
        #     self.load_image()
        label = tk.Label(self.left_frame, text="Input", font=("Arial", 14))
        label.pack(anchor="w", padx=10, pady=10)

        # Placeholder for input image
        # self.input_canvas = tk.Label(self.left_frame, text="[Input Image]", bg="lightblue", width=40, height=20)
        self.input_canvas.pack(padx=10, pady=10)

    def build_center(self):
        buttons = ["Load", "Rectangle Mask", "Segment", "Save", "Next step"]
        bt_action = [lambda: self.load_mask(), lambda: self.direct_draw() ,lambda: self.run_segment(),lambda: self.save_mask(), lambda: self.next_step()] # using lambda: ..., delay execution — the code inside runs only when the button is clicked. 
        for text,cmd in zip(buttons,bt_action):
            btn = tk.Button(self.center_frame, text=text,command=cmd, width=15)
            btn.pack(pady=10)

    def build_right(self):
        label = tk.Label(self.right_frame, text="Mask", font=("Arial", 14))
        label.pack(anchor="w", padx=10, pady=10)

        # Placeholder for mask image
        # self.mask_canvas = tk.Label(self.right_frame, text="[Mask Image]", bg="lightcoral", width=40, height=20)
        # self.mask_canvas.pack(padx=10, pady=10)
        self.seg_canvas.pack(padx=10, pady=10)

    def build_header(self):
        # Configure 3 columns in grid
        self.header_frame.grid_columnconfigure(0, weight=1)
        self.header_frame.grid_columnconfigure(1, weight=1)
        self.header_frame.grid_columnconfigure(2, weight=1)
        self.btn_left = tk.Button(self.header_frame, text='<',command=lambda: self.previous_image(), width=15)
        self.btn_left.grid(row=0, column=0, sticky="w")
        side_label = tk.Label(self.header_frame, textvariable=self.side_label_show, font=("Arial", 14))
        side_label.grid(row=0, column=1)
        self.btn_right = tk.Button(self.header_frame, text='>',command=lambda: self.next_image(), width=15)
        self.btn_right.grid(row=0, column=2, sticky="e")



# ------------------ precessing page --------------------------
class ImageProcessingApp(tk.Frame):
    def __init__(self,parent,controller,configs):
        super().__init__(parent, bg="green")
        self.page_ctrl = controller
        self.config_data = {}  # <-- This will store the current config
        self.configs = configs
        # self.coor = 0
        # self.mask = ""
        # self.img_top_h = 0
        # self.img_top_w = 0
        # self.img_pano_h =  0
        # self.img_pano_w = 0
        # self.seamless_center_h = 0
        # self.seamless_center_w = 0
        # self.move_img_up = 0
        # self.move_img_down = 0
        # self.x_new_coor = [0, 0]
        # self.y_new_coor = [0,0]
        
        # สร้าง Frame หลักที่มีกรอบครอบ UI ทั้งหมด
        main_frame = tk.Frame(self, bd=0.5, relief="solid", highlightbackground="green", highlightcolor="green", highlightthickness=0.5)
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # self.title("Inpaint System Ver.1.3")
        # self.geometry("700x550")
        # icon_path = r"C:/Users/batma/OneDrive/Documents/inpaint-system/cmd-inpaint/icon.ico"
        # icon_path = "icon.ico"
        # self.iconbitmap(get_file_path(icon_path))
        
        # ตั้งค่า font หลักสำหรับทุก widget เป็น Kanit Regular 10
        self.option_add("*Font", ("Kanit", 10))

        # Frame สำหรับเลือกโฟลเดอร์
        dir_frame = tk.Frame(main_frame)
        dir_frame.pack(fill="x", pady=(20, 5))

        tk.Label(dir_frame, text="เลือก Folder :").pack(side="left", padx=5)
        self.dir_entry = tk.Entry(dir_frame, width=60)
        self.dir_entry.pack(side="left", padx=5)
        tk.Button(dir_frame, text="เลือก", command=self.browse_dir, bg="lightblue", fg="black").pack(side="left", padx=5)

        # Frame สำหรับเลือกโฟลเดอร์
        config_status_frame = tk.Frame(main_frame)
        config_status_frame.pack(fill="x", pady=5)

        tk.Label(config_status_frame, text="เลือก Config : ").pack(side="left", padx=5)

        self.message_config = tk.StringVar()
        self.message_config.set('')

        # Dropdown
        self.selected_config = tk.StringVar()
        self.dropdown = ttk.Combobox(config_status_frame, textvariable=self.selected_config, values=list(self.configs.keys()), state="readonly")
        self.dropdown.pack(side="left", padx=5)
        self.selected_config.set(list(self.configs.keys())[0])
        # self.set_initial_config(list(self.configs.keys())[0])
        # Bind the selection event
        self.dropdown.bind("<<ComboboxSelected>>", self.refresh_config)
        self.dropdown.bind("<Button-1>", self.refresh_file_config)
        self.switch_button = tk.Button(config_status_frame, text="mask รถ", command=lambda: self.start_segment(), bg="white", fg="black", state=tk.DISABLED)
        self.switch_button.pack(side="left", padx=15)

        # Display area
        tk.Label(config_status_frame,textvariable = self.message_config).pack(side="left", padx=5)

        # Frame สำหรับ Radiobutton
        dir_frame2 = tk.Frame(main_frame)
        dir_frame2.pack(fill="x", pady=5)

        tk.Label(dir_frame2, text="ประมวลผลภาพใหม่ทั้งหมด :").pack(side="left", padx=5, anchor="w")
        
        # ตัวแปรสำหรับ Radiobutton
        self.overwrite_var = tk.StringVar(value="False")
        
        tk.Radiobutton(dir_frame2, text="ใช่ (True)", variable=self.overwrite_var, value="True").pack(side="left", padx=5)
        tk.Radiobutton(dir_frame2, text="ไม่ใช่ (False)", variable=self.overwrite_var, value="False").pack(side="left", padx=5)

        # Run Button
        self.run_button = tk.Button(dir_frame2, text="เริ่มประมวลผล", command=lambda: self.start_thread_image(), bg="green", fg="white")
        self.run_button.pack(side="left", padx=15)

        # switch Button
        # dir_frame_switch = tk.Frame(main_frame)
        # dir_frame_switch.pack(fill="x", pady=5)
        # self.switch_button = tk.Button(dir_frame_switch, text="switch", command=lambda: controller.show_frame(PageTwo), bg="pink", fg="white")
        # self.switch_button.pack(side="left", padx=15)
        # tk.Label(dir_frame_switch, textvariable=controller.shared_data["input_text"]).pack(side="left", padx=5, anchor="w")

        # cancle Button
        self.cancle_button = tk.Button(dir_frame2, text="ยกเลิก", command=lambda: self.stop_process(), bg="red", fg="white")
        self.cancle_button.pack(side="left", padx=15)
        self.hide_button(self.cancle_button)

        # Frame สำหรับ Progress
        dir_frame3 = tk.Frame(main_frame)
        dir_frame3.pack(pady=20, padx=5, anchor="c")

        # Progress bar widget
        self.progress_per = tk.IntVar()
        self.progress = tk.StringVar()
        self.progress_folder = tk.StringVar()
        self.progress_folder.set('')
        self.progress.set('0/0')
        tk.Label(dir_frame3,textvariable = self.progress_folder).pack(side="left")
        self.progressbar = ttk.Progressbar(dir_frame3,variable=self.progress_per,length=400).pack(side="left")
        # self.progressbar.pack(pady=20)
        tk.Label(dir_frame3,textvariable = self.progress).pack(side="left")
        # self.progress = 

        # แสดงภาพ
        image_path = "flow.png"
        # image_path = r"C:/Users/batma/OneDrive/Documents/inpaint-system/cmd-inpaint/flow.png"
        
        try:
            image = Image.open(get_file_path(image_path))
            image = image.resize((450, 250))
            self.photo = ImageTk.PhotoImage(image)
            tk.Label(main_frame, image=self.photo).pack(pady=10)
        except Exception as e:
            print(f"Error loading image: {e}")

        # ข้อความด้านล่างสุด
        tk.Label(main_frame, text="โครงการพัฒนาแบบจำลองภูมิประเทศความละเอียดสูงเพื่อการบริหารจัดการน้ำ และแก้ไขปัญหาอุทกภัยอย่างยั่งยืน",
                 wraplength=480, justify="center", fg="green").pack(side="bottom", pady=10)
        
        # self.over_class = OverlayGPS(self.config_data,"kmutnb")
        self.set_initial_config(self.selected_config.get())

    def start_segment(self):
        directory = self.dir_entry.get()
        path_input = find_fol(directory)
        if(len(path_input) > 0):
            dir_list_input = glob.glob(path_input[0]+'*'+'.jpg')
            self.page_ctrl.shared_data["input_img"] = dir_list_input[0]
            self.page_ctrl.shared_data["config_data"] = self.config_data
            self.page_ctrl.show_frame(PageTwo)
        else:
            print('ไม่มีภาพในโฟลเดอร์')

        # # result_tk = ImageTk.PhotoImage(result)
        # label = tk.Label(top, image=result)
        # label.image = result  # Keep a reference!
        # label.pack()

    def state_setting(self,config_name):
        self.message_config.set('Start read '+config_name+' config...')
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
        self.over_class = OverlayGAN(self.config_data,config_name)
        generator = set_model(self.config_data['model_type'],self.config_data["model_input"],self.config_data["model_weight"])
        self.over_class.set_model(generator)
        self.over_cube_class = OverlayCube(self.config_data,config_name)
        self.over_cube_class.set_model(generator)
        # self.over_cube_class.check_data()
        self.message_config.set('Complete setting '+config_name)
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที

    def set_initial_config(self, config_name):
        if config_name in self.configs:
            self.selected_config.set(config_name)
            self.config_data = self.configs.get(config_name, {})  #  SET VALUE HERE
            self.state_setting(config_name)
        else:
            self.selected_config.set(list(self.configs.keys())[0])
            # config_name = self.selected_config.get()
            self.config_data = self.configs.get(list(self.configs.keys())[0], {})  #  SET VALUE HERE
            self.state_setting((self.configs.keys())[0])
    
    # Refresh content when dropdown changes
    def refresh_config(self,event=None):
        # self.refresh_file_config()
        config_name = self.selected_config.get()
        self.config_data = self.configs.get(config_name, {})  # 🔥 SET VALUE HERE
        # print(self.config_data)
        # self.output.delete("1.0", tk.END)
        # self.output.insert(tk.END, json.dumps(self.config_data["coor"], indent=2))
        # self.over_class = OverlayGPS(self.config_data,config_name)
        self.state_setting(config_name)
    
    def refresh_file_config(self,event):
        # self.message_config.set('update config file...')
        config_path = get_file_path("config_gan_ui.json")
        with open(config_path, "r") as f:
            self.configs = json.load(f)
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
        # self.message_config.set('update complete')
        self.dropdown["values"] = list(self.configs.keys())
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
        # print('update: '+str(list(self.configs.keys())))

    def browse_dir(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, folder_selected)
            self.run_button.config(text="เริ่มประมวลผล", bg="green")
            self.switch_button.config(state=tk.NORMAL)
            self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที

    def hide_button(self,widget):
        # This will remove the widget from toplevel
        widget.pack_forget()


    # Method to make Button(widget) visible
    def show_button(self,widget):
        # This will recover the widget from toplevel
        widget.pack(side="left", padx=15)

    
    def cancel_action(self):
        self.run_button.config(text="เริ่มประมวลผล", bg="green")
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที

    def stop_process(self):
        print('stop process thread')
        event_obj.set()
        self.hide_button(self.cancle_button)
        # self.cancel_action()

    def run_process(self):
        directory = self.dir_entry.get()
        overwrite = self.overwrite_var.get()
        self.run_button.config(text="กำลังประมวลผล", bg="orange")
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที

        if not directory:
            messagebox.showerror("Error", "Please select a directory.")
            return
        
        self.start_processing(directory, overwrite)

    def start_thread_image(self):
        """Starts a new thread for the long-running task."""
        print('start thread')
        # if(self.page_ctrl.shared_data["segment_mask"] != None):
        if not all(value is None for value in self.page_ctrl.shared_data["segment_mask"].values()):
            if(self.page_ctrl.shared_data["segment_mask"]["top_view"] is not None):
                self.over_class.set_segMask(self.page_ctrl.shared_data["segment_mask"]["top_view"])
            keys_to_check = ["cube_front","cube_right","cube_back","cube_left","cube_top","cube_bottom"]
            if not all(self.page_ctrl.shared_data["segment_mask"][key] is None for key in keys_to_check):
                # Get keys as a list
                keys = list(self.page_ctrl.shared_data["segment_mask"].keys())

                # Define range
                start = keys.index("cube_front")
                end = keys.index("cube_bottom")

                # Slice and build new dictionary
                selected = {k: self.page_ctrl.shared_data["segment_mask"][k] for k in keys[start:end + 1]}
                self.over_cube_class.set_segMask(selected)
        directory = self.dir_entry.get()
        overwrite = self.overwrite_var.get()
        self.run_button.config(text="กำลังประมวลผล", bg="orange")
        # self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
        self.show_button(self.cancle_button)
        # self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
        self.progress.set('0/0')
        # self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
        self.progress_folder.set('')
        self.progress_per.set(0)
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที

        if not directory:
            messagebox.showerror("Error", "Please select a directory.")
            return
        # ---------- start inpainting thread ----------------------------
        thread = threading.Thread(target=self.start_processing, args=(directory,overwrite))
        thread.daemon = True  # Daemon thread will exit when the main program exits
        thread.start()
        # --------------------------------------------------------------

    def start_processing(self, directory, overwrite):
        event_obj.clear()
        # Placeholder function for integrating your processing logic
        print(f"Directory: {directory}")
        print(f"Overwrite: {overwrite}")     
        path_input = find_fol(directory)
        path_input.sort()
        print("all path_input: ")
        for print_path in path_input: # show all album
            print(os.path.basename(os.path.dirname(os.path.dirname(print_path))))
 
        for i in range(len(path_input)): # run through each album
            if event_obj.is_set():
                self.run_button.config(text="หยุดการทำงาน", bg="red")
                self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
                break
            print(f"path_input: {path_input[i]}")
            
            # ดึงชื่อโฟลเดอร์ที่อยู่ก่อนสุดท้าย
            line_name = os.path.basename(os.path.dirname(os.path.dirname(path_input[i])))
            print(f"line_name: {line_name}")
            self.progress_folder.set(line_name)
            self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
            
            save_path = path_input[i] + line_name + '_output'
            
            main_path = path_input[i]
            # print(main_path)
            target = []
            dir_list_input = glob.glob(main_path+'*'+'.jpg')
            dir_list_input.sort()
            if(not os.path.exists(save_path)): # if not found output folder -> create folder
                create_fol(save_path)
            else:
                if(not bool(strtobool(overwrite))): # if not overwrite file
                    for item in dir_list_input[:]: 
                        pic_full_name = (item.replace('\\','/')).split('/')[-1]
                        pic_name = save_path+'/'+pic_full_name[:-4]+'_O.jpg'
                        if(os.path.exists(pic_name)):
                            dir_list_input.remove(item)
            if(len(dir_list_input)>0):
                for i in range(len(dir_list_input)):
                    # print(dir_list_input[i])
                    # print(dir_list_input[i].split('\\'))
                    name_file = ((dir_list_input[i].replace('\\','/')).split('/'))[-1]
                    # print(name_file)
                    num_file = ((name_file.split('_'))[1])[:-4]
                    target.append(int(num_file))
            
                for count_pic, coor in enumerate(target):
                    if event_obj.is_set():
                        self.run_button.config(text="หยุดการทำงาน", bg="red")
                        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
                        break
                    # ---- inpaint image ver. gps ---------------------------------
                    # source_pic = self.over_class.find_relate_new(coor_detail,coor,"utm")
                    # self.over_class.overlay(main_path,save_path,source_pic,coor,str(coor),coor_detail)
                    # --------------------------------------------------------
                    # end inpaint process file
                    keys_to_check = ["cube_front","cube_right","cube_back","cube_left","cube_top","cube_bottom"]
                    if(self.page_ctrl.shared_data["segment_mask"]["top_view"] is not None):
            #     self.over_class.set_segMask(self.page_ctrl.shared_data["segment_mask"]["top_view"])
            # keys_to_check = ["cube_front","cube_right","cube_back","cube_left","cube_top","cube_bottom"]
            # if not all(self.page_ctrl.shared_data["segment_mask"][key] is None for key in keys_to_check):
                        final_pano_output = self.over_class.overlay(main_path,save_path,coor)
                        if not all(self.page_ctrl.shared_data["segment_mask"][key] is None for key in keys_to_check):
                            self.over_cube_class.overlay(final_pano_output,save_path,coor)
                    else:
                        if not all(self.page_ctrl.shared_data["segment_mask"][key] is None for key in keys_to_check):
                            self.over_cube_class.overlay(main_path,save_path,coor)
                    percent_num = (int(count_pic+1)*100)/len(target)
                    # print(count_pic,' from ',len(target),' => ',percent_num,'%')
                    self.progress.set(str(count_pic+1)+'/'+str(len(target)))
                    self.progress_per.set(int(percent_num))
                    self.update_idletasks()
                    # time.sleep(1) 
        if not(event_obj.is_set()):
            self.run_button.config(text="เสร็จสิ้น", bg="lightgreen")
            self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
            self.hide_button(self.cancle_button)
            self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
               

# Run the Application
if __name__ == "__main__":
    # Load configs
    config_path = get_file_path("config_gan_ui.json")
    with open(config_path, "r") as f:
        configs = json.load(f)
    # print(configs["kmutnb"])
    # initial_config = configs["kmutnb"]
    app = MainApp(configs)
    # app = ImageProcessingApp(configs)
    app.mainloop()
