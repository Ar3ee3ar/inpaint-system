import tkinter as tk
from tkinter import font
from tkinter import filedialog, messagebox, ttk
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

# import matplotlib.pyplot as plt

event_obj = threading.Event()

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
        end_proj = time.time()
        # print('proj: ',end_proj - start_proj)

        # car image
        per_img = crop_center(per_img_1000,(self.model_input[0],self.model_input[1])) # resize to 290,190
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
        predict_image_norm = pad_images_to_same_size(predict_image_norm)
        # predict_image_norm = ((1 - mask_img_pano) * per_img_1000) + (mask_img_pano * predict_image_norm)
        predict_image_norm = self.seamless((predict_image_norm*255.0).astype(np.uint8),(per_img_1000*255.0).astype(np.uint8))
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
    #                 pic_full_name = item.split('\\')[-1]
    #                 pic_name = save_path+'/'+pic_full_name[:-4]+'_O.jpg'
    #                 if(os.path.exists(pic_name)):
    #                     dir_list_input.remove(item)
    #     if(len(dir_list_input)>0):
    #         for i in range(len(dir_list_input)):
    #             name_file = (dir_list_input[i].split('\\'))[1]
    #             num_file = ((name_file.split('_'))[1])[:-4]
    #             target.append(int(num_file))
        
    #         for coor in tqdm(target):
    #             source_pic = self.find_relate_new(coor_detail,coor,"utm")
    #             self.overlay(main_path,save_path,source_pic,coor,str(coor),coor_detail)

# class MainApp(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Inpaint System Ver.1.4")
#         self.geometry("700x550")

#         # container holds all pages
#         container = tk.Frame(self)
#         container.pack(Fill="both, expand=True")

#         # store pages in a dict
#         self.frames = {}


# Tkinter UI Class
class ImageProcessingApp(tk.Tk):
    def __init__(self,configs):
        super().__init__()
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
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        self.title("Inpaint System Ver.1.3")
        self.geometry("700x550")
        # icon_path = r"C:\Users\batma\OneDrive\Documents\inpaint-system\cmd-inpaint\icon.ico"
        icon_path = "icon.ico"
        self.iconbitmap(get_file_path(icon_path))
        
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
        # image_path = r"C:\Users\batma\OneDrive\Documents\inpaint-system\cmd-inpaint\flow.png"
        
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
        top = tk.Toplevel(self.root)
        top.title("Segmentation process")
        # # result_tk = ImageTk.PhotoImage(result)
        # label = tk.Label(top, image=result)
        # label.image = result  # Keep a reference!
        # label.pack()

    def state_setting(self,config_name):
        self.message_config.set('Start read '+config_name+' config...')
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
        self.over_class = OverlayGAN(self.config_data,config_name)
        self.message_config.set('Complete setting '+config_name)
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที

    def set_initial_config(self, config_name):
        if config_name in self.configs:
            self.selected_config.set(config_name)
            self.config_data = self.configs.get(config_name, {})  # 🔥 SET VALUE HERE
            self.state_setting(config_name)
        else:
            self.selected_config.set(list(self.configs.keys())[0])
            # config_name = self.selected_config.get()
            self.config_data = self.configs.get(list(self.configs.keys())[0], {})  # 🔥 SET VALUE HERE
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
        # thread = threading.Thread(target=self.start_processing, args=(directory,overwrite))
        # thread.daemon = True  # Daemon thread will exit when the main program exits
        # thread.start()
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
                        pic_full_name = item.split('\\')[-1]
                        pic_name = save_path+'/'+pic_full_name[:-4]+'_O.jpg'
                        if(os.path.exists(pic_name)):
                            dir_list_input.remove(item)
            if(len(dir_list_input)>0):
                for i in range(len(dir_list_input)):
                    name_file = (dir_list_input[i].split('\\'))[1]
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
                    self.over_class.overlay(main_path,save_path,coor)
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
    app = ImageProcessingApp(configs)
    app.mainloop()
