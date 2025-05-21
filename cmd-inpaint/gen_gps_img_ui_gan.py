import tkinter as tk
from tkinter import font
from tkinter import filedialog, messagebox, ttk
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
# tool
# from .tools.utils import generate_images, wandb_log, view_test
# from .tools.loss import discriminator_loss,generator_loss
from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from tools.process_pano import Equirectangular,Perspective,inpaint_pano
# from initial_gps import *
from initial_gan import image_size,width,height,in_mask,generator, mask_pano_img, mask_img_full,mask_img_seamless
import os
import glob
import time
import threading

event_obj = threading.Event()


def create_fol(path):
    if not os.path.exists(path): 
        os.makedirs(path)
    else:
        print('already create '+path)

def save_img(path, inpaint_img):
    # save inpaint image
    print('save')
    impainted_image_im = Image.fromarray(((inpaint_img)*255.0).astype(np.uint8))
    impainted_image_im.save(path)


# start tools for gan process ------------------------------------------------
def seamless_gan(source,target):
    mask = mask_img_seamless
    (w,h) = mask.shape[:2]
    center = (w//2,h//2)
    # center = (499,450)
    im_clone = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
    return im_clone

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
    per_img_1000 = equ.GetPerspective(167, 0, -90, 1000, 1000)  # Specify parameters(FOV, theta, phi, height, width)
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
    predict_image_norm = seamless_gan((predict_image_norm*255.0).astype(np.uint8),(per_img_1000*255.0).astype(np.uint8))
    predict_image_norm = predict_image_norm/255.0
    end_pro_img = time.time()
    print('overlay: ',end_pro_img - start_pro_img)

    start_reproj = time.time()
    equ = Equirectangular(predict_image_norm,167, 0, -90)
    inpaint_pano_img,_ = equ.GetEquirec(height,width)
    end_reproj = time.time()
    print('reproj: ',(end_reproj - start_reproj)) 

    inpaint_fill = inpaint_pano(inpaint_pano_img*255.0,mask_pano_img,car_pano_img)
    # inpaint_mask = inpaint_pano(mask_pano_img,mask_pano_img,car_pano_img)
    end_pano = time.time()

    # if(output_mode=="save"):
    save_path = save_path +  "/img_"+str(target_coor).rjust(6, '0')+"_O.jpg"
    print(save_path)
    start_save = time.time()
    save_img(path=save_path,inpaint_img=inpaint_fill)
    end_save = time.time()
    print('save_time: ',(end_save - start_save))
    all_time = all_time + (end_pano - start_pano)
    print('all_time: ',all_time)
# end tools for gps process --------------------------------------------------   

def find_fol(main_input):
    dir_list_input = glob.glob(main_input+'/**/'+'Camera/',recursive=True)
    for i in range(len(dir_list_input)):
        dir_list_input[i] = dir_list_input[i].replace('\\','/')
    return dir_list_input

# Tkinter UI Class
class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # สร้าง Frame หลักที่มีกรอบครอบ UI ทั้งหมด
        main_frame = tk.Frame(self, bd=0.5, relief="solid", highlightbackground="green", highlightcolor="green", highlightthickness=0.5)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        self.title("Inpaint System Ver.2.0")
        self.geometry("700x550")
        # icon_path = r"C:\Users\batma\OneDrive\Documents\inpaint-system\cmd-inpaint\icon.ico"
        icon_path = "icon.ico"
        self.iconbitmap(icon_path)
        
        # ตั้งค่า font หลักสำหรับทุก widget เป็น Kanit Regular 10
        self.option_add("*Font", ("Kanit", 10))

        # Frame สำหรับเลือกโฟลเดอร์
        dir_frame = tk.Frame(main_frame)
        dir_frame.pack(fill="x", pady=(20, 5))

        tk.Label(dir_frame, text="เลือก Folder :").pack(side="left", padx=5)
        self.dir_entry = tk.Entry(dir_frame, width=60)
        self.dir_entry.pack(side="left", padx=5)
        tk.Button(dir_frame, text="เลือก", command=self.browse_dir, bg="lightblue", fg="black").pack(side="left", padx=5)

        # Frame สำหรับ Radiobutton
        dir_frame2 = tk.Frame(main_frame)
        dir_frame2.pack(fill="x", pady=5)

        tk.Label(dir_frame2, text="ประมวลผลภาพใหม่ทั้งหมด :").pack(side="left", padx=5, anchor="w")
        
        # ตัวแปรสำหรับ Radiobutton
        self.overwrite_var = tk.StringVar(value="False")
        
        tk.Radiobutton(dir_frame2, text="ใช่ (True)", variable=self.overwrite_var, value="True").pack(side="left", padx=5)
        tk.Radiobutton(dir_frame2, text="ไม่ใช่ (False)", variable=self.overwrite_var, value="False").pack(side="left", padx=5)

        # Run Button GAN
        self.run_button = tk.Button(dir_frame2, text="เริ่มประมวลผล GAN ", command=lambda: self.start_thread_image('GAN'), bg="green", fg="white")
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
            image = Image.open(image_path)
            image = image.resize((450, 250))
            self.photo = ImageTk.PhotoImage(image)
            tk.Label(main_frame, image=self.photo).pack(pady=10)
        except Exception as e:
            print(f"Error loading image: {e}")

        # ข้อความด้านล่างสุด
        tk.Label(main_frame, text="โครงการพัฒนาแบบจำลองภูมิประเทศความละเอียดสูงเพื่อการบริหารจัดการน้ำ และแก้ไขปัญหาอุทกภัยอย่างยั่งยืน",
                 wraplength=480, justify="center", fg="green").pack(side="bottom", pady=10)

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

    def start_thread_image(self, method):
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
        # if(method == 'GPS'):
        #     thread = threading.Thread(target=self.start_processing, args=(directory,overwrite))
        if(method == 'GAN'):
            thread = threading.Thread(target=self.start_processing_gan, args=(directory,overwrite))
        thread.daemon = True  # Daemon thread will exit when the main program exits
        thread.start()
    
    def start_processing_gan(self, directory, overwrite):
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
                    # source_pic = find_relate_new(coor_detail,coor,"utm")
                    # overlay(main_path,save_path,source_pic,coor,str(coor),coor_detail)
                    # --------------------------------------------------------
                    # end inpaint process file
                    main_gan_nakorn(main_path,save_path,coor)
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
    app = ImageProcessingApp()
    app.mainloop()
