import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import glob
import json
import threading
import os
from distutils.util import strtobool
import sys
from tensorflow import keras
import tensorflow as tf
import torch

# from .mask_page import MaskApp

sys.path.append("..")
from tools import find_file_path
from tools.utils import get_file_path, find_fol, create_fol
from tools.overlay import OverlayCube, OverlayGAN
from tools.pix2pix import Generator as p2pG
from tools.pix2pix import Discriminator as p2pD
from tools.deepfill.inpaint_model import InpaintGenerator
from tools.sam_tools.segment_anything import sam_model_registry


event_obj = threading.Event()

# ----------------- initial model setting ---------------
def load_model_generator(model_type,model_input_2d,model_weight):
    print('start inpainting model')
    # initial generator model
    keras.backend.clear_session()
    # yolo_model = YOLO("yolov8n.pt")
    if(model_type == "p2p"):
        generator = p2pG.Generator(input_shape=[model_input_2d[0],model_input_2d[1],3])
        generator.load_weights(find_file_path(model_weight)) # 500
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
        generator.load_weights(find_file_path(model_weight))
    print('set_model: ',model_type)
    print('set_model: ',generator)
    print('inpainting model start success')
    return generator

def load_model_segment():
    print('start segmentation model')
    sam = sam_model_registry["vit_l"](checkpoint=find_file_path("weight_sam/sam_vit_l_0b3195.pth")).to(device=torch.device('cpu'))
    print('segmentation model start success')
    return sam

# ------------------ precessing page --------------------------
class ImageProcessingApp(tk.Frame):
    def __init__(self,parent,controller,configs):
        super().__init__(parent, bg="green")
        self.page_ctrl = controller
        self.config_data = {}  # <-- This will store the current config
        self.configs = configs
        self.type_process_scale = 'resize' # 'pad_image' or 'resize'
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
        # self.switch_button = tk.Button(dir_frame_switch, text="switch", command=lambda: controller.show_frame(MaskApp), bg="pink", fg="white")
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
        image_path = "image_app/flow.png"
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
            self.page_ctrl.shared_data["type_process_scale"] = self.type_process_scale
            self.page_ctrl.show_frame('MaskApp')
        else:
            print('ไม่มีภาพในโฟลเดอร์')

        # # result_tk = ImageTk.PhotoImage(result)
        # label = tk.Label(top, image=result)
        # label.image = result  # Keep a reference!
        # label.pack()

    def state_setting(self,config_name,status_setting):
        self.message_config.set('Start read '+config_name+' config...')
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที
        self.over_class = OverlayGAN(self.config_data,config_name,self.type_process_scale)
        generator = load_model_generator(self.config_data['model_type'],self.config_data["model_input"],self.config_data["model_weight"])
        self.over_class.set_model_generator(generator)
        self.over_cube_class = OverlayCube(self.config_data,config_name,self.type_process_scale)
        self.over_cube_class.set_model_generator(generator)
        if(status_setting == 'initial'):
            segment_model = load_model_segment()
            self.page_ctrl.shared_data['segment_model'] = segment_model
            # self.over_class.set_model_segment(segment_model)
            # self.over_cube_class.set_model_segment(segment_model)
        # self.over_cube_class.check_data()
        self.message_config.set('Complete setting '+config_name)
        self.update_idletasks()  # อัปเดต UI ให้เห็นการเปลี่ยนแปลงทันที

    def set_initial_config(self, config_name):
        if config_name in self.configs:
            self.selected_config.set(config_name)
            self.config_data = self.configs.get(config_name, {})  #  SET VALUE HERE
            self.state_setting(config_name,'initial')
        else:
            self.selected_config.set(list(self.configs.keys())[0])
            # config_name = self.selected_config.get()
            self.config_data = self.configs.get(list(self.configs.keys())[0], {})  #  SET VALUE HERE
            self.state_setting((self.configs.keys())[0],'initial')
    
    # Refresh content when dropdown changes
    def refresh_config(self,event=None):
        # self.refresh_file_config()
        config_name = self.selected_config.get()
        self.config_data = self.configs.get(config_name, {})  # 🔥 SET VALUE HERE
        # print(self.config_data)
        # self.output.delete("1.0", tk.END)
        # self.output.insert(tk.END, json.dumps(self.config_data["coor"], indent=2))
        # self.over_class = OverlayGPS(self.config_data,config_name)
        self.state_setting(config_name,'update')
    
    def refresh_file_config(self,event):
        # self.message_config.set('update config file...')
        config_path = find_file_path("config_gan_ui.json")
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
                        pic_full_name = item.split('\\')[-1]
                        pic_name = save_path+'/'+pic_full_name[:-4]+'_O.jpg'
                        if(os.path.exists(pic_name)):
                            dir_list_input.remove(item)
            if(len(dir_list_input)>0):
                for i in range(len(dir_list_input)):
                    # print(dir_list_input[i])
                    # print(dir_list_input[i].split('\\'))
                    name_file = (dir_list_input[i].split('\\'))[-1]
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
               
