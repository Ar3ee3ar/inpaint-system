import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from tkinter import filedialog, messagebox, ttk, Menu, Menubutton, simpledialog
import numpy as np
import cv2
import sys
import json
import threading
import os
# from .main_page import ImageProcessingApp
from .loadingPopup_page import LoadingPopup
from . import find_file_path, get_file_path

sys.path.append("..")
from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from tools.process_pano import Equirectangular,Perspective,inpaint_pano
from tools.utils import up_downscale, DictNavigator, binary_change
from tools.py360convert.py360convert import c2e, e2c
from tools.sam_tools.classification_torch import bbox
# --------------------- segment page ------------------------

def edit_json_by_key(file_path, key_to_match, new_value,type_command):
    # edit_flag = False
    try:
        with open(find_file_path(file_path), 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # print(f"Error: File not found at {file_path}")
        messagebox.showerror("Error", f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        # print(f"Error: Invalid JSON format in {file_path}")
        messagebox.showerror("Error", f"Error: Invalid JSON format in {file_path}")
        return
    
    if(type_command == 'update'):
        for key, value in data.items():
            if key == key_to_match:
                data[key] = new_value
                # edit_flag = True
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4) # Optional: indent for pretty formatting
                messagebox.showinfo("Info", "Config file updated successfully.")
    elif(type_command == 'add'):
        currentKey_list = list(data.keys())
        if(key_to_match not in currentKey_list):
            data[key_to_match] = new_value
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4) # Optional: indent for pretty formatting
            messagebox.showinfo("Info", "Add new config file successfully.")
        else:
            messagebox.showerror("Error", f"Already have config file name: {key_to_match}")
            return

    # with open(file_path, 'w') as file:
    #     json.dump(data, file, indent=4) # Optional: indent for pretty formatting
    # print("JSON file updated successfully.")

class MaskApp(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bd=0.5, relief="solid", highlightbackground="green", highlightcolor="green", highlightthickness=0.5)
        self.parent_frame = parent
        self.page_ctrl = controller
        # ------------ variable for segment task
        self.config_data = self.page_ctrl.shared_data["config_data"]
        self.show_side_name = {
                "top_view": "Bird view",
                "cube_front": "Front view",
                "cube_right": "Right view",
                "cube_back": "Back view",
                "cube_left": "Left view",
                "cube_top": "Upper view",
                "cube_bottom": "Bottom view"
            }
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
        self.rect = None # rectangle paint from input section
        self.mask_canvas = None
        self.mask_canvas_draw = None
        self.seg_mask_canvas = None
        self.seg_mask_canvas_draw = None
        self.binary_mask = {
                "top_view": None,
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        blank_mask = np.full((self.config_data['img_top_h'],self.config_data['img_top_w'],3), 0.0, np.float32)
        self.blank_mask = up_downscale(blank_mask,(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
        self.start_x,self.start_y,self.end_x,self.end_y = None, None, None, None
        # if(len(self.config_data)>0):
        # ----------- create element in frame --------------------------
        # tk.Label(self, text="Input", font=("Arial", 18)).pack(pady=20)
        self.header_frame = tk.Frame(self, width=600, bg="green", height=50)
        self.header_frame.pack(fill="both")
        self.side_label_key = tk.StringVar()
        self.side_label_show = tk.StringVar()
        self.nav_image = None
        self.btn_left = None
        self.btn_right = None
        self.seg_progress = ttk.Progressbar(self, mode='indeterminate')
        # tk.Label(self, textvariable=controller.shared_data["input_text"]).pack()
        # tk.Button(self, text="Go to Page One", command=lambda: controller.show_frame(ImageProcessingApp)).pack()
        # Create main 3-part horizontal layout
        self.left_frame = tk.Frame(self, width=300, bg="white", bd=0.5, relief="solid", highlightbackground="green", highlightcolor="green", highlightthickness=0.5)
        self.menubar = Menu(self)
        # self.center_frame = tk.Frame(self, width=100, bg="lightgray")
        self.right_frame = tk.Frame(self, width=300, bg="white" , bd=0.5, relief="solid", highlightbackground="green", highlightcolor="green", highlightthickness=0.5)

        self.left_frame.pack(side="left", fill="both", expand=False)
        # self.center_frame.pack(side="left", fill="y")
        self.right_frame.pack(side="left", fill="both", expand=False)

        self.input_canvas = tk.Canvas(self.left_frame)
        self.input_canvas_image = None

        self.seg_canvas = tk.Canvas(self.right_frame)
        self.seg_canvas_image = None

        self.build_left()
        # self.build_toolBar()
        # self.build_center()
        self.build_right()
        self.build_header()

        self.input_canvas.bind("<ButtonPress-1>", self.draw_rec_press)
        self.input_canvas.bind("<B1-Motion>", self.draw_rec_move)
        self.input_canvas.bind("<ButtonRelease-1>", self.draw_rec_release)
        self.input_canvas.bind("<Button-3>",self.clear_rec)

        self.seg_canvas.bind("<B1-Motion>", self.paint)
        # if(self.page_ctrl.shared_data['input_img'] != None):
        #     self.load_image()

    def start_thread_on_show(self):
        self.loading = LoadingPopup(self,"Processing...")
        threading.Thread(target = self.on_show).start()

    def on_show(self):
        # print(self.page_ctrl.shared_data["config_name"])
        # print(self.page_ctrl.shared_data["config_data"])
        self.config_data = self.page_ctrl.shared_data["config_data"]
        self.pre_process_img()
        if(not self.page_ctrl.shared_data["mask_edited"]):
            reset_segment_mask = {key: None for key in self.binary_mask}
            self.binary_mask = reset_segment_mask
            self.pre_process_mask()
        self.set_canvas(list(self.image.keys())[0])
        self.after(0, self.loading.close) # Close popup safely from main thread

    def start_thread_pre_process_img(self):
        self.loading = LoadingPopup(self, "Processing...")
        threading.Thread(target=self.pre_process_img).start()

    def pre_process_img(self):
        # print(self.config_data)
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
        # per_img = crop_center(self.full_top_img,(self.config_data['model_input'][0],self.config_data['model_input'][1])) # crop (downscale) to match model input size
        per_img = up_downscale(self.full_top_img,(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
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
                # car_cube_img = crop_center(self.image_pano_cube[count],(self.config_data['model_input'][0],self.config_data['model_input'][1])) # crop (downscale) to match model input size
                car_cube_img = up_downscale(self.image_pano_cube[count],(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
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
        # self.after(0, self.loading.close)  # Close popup safely from main thread
    
    def start_thread_pre_process_mask(self):
        self.loading = LoadingPopup(self, "Processing...")
        threading.Thread(target=self.pre_process_mask).start()

    def pre_process_mask(self):
        for count, side_name in enumerate(self.binary_mask):
            file_path = self.config_data["mask"][side_name]
            if(file_path != ""):
                self.binary_mask[side_name] = preprocess_img(find_file_path(file_path),(self.config_data['img_top_h'], self.config_data['img_top_w'])) # open file by pillow
                self.binary_mask[side_name] = binary_change(self.binary_mask[side_name])
                # self.binary_mask[self.side_label_key.get()] = crop_center(self.binary_mask[self.side_label_key.get()],(self.config_data['model_input'][0],self.config_data['model_input'][1])) # crop (downscale) to match model input size
                self.binary_mask[side_name] = up_downscale(self.binary_mask[side_name],(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
                # overlay_image = ((self.binary_mask[side_name]) * np.array(self.image[side_name])) + np.array(self.image[side_name])
                overlay_image = np.where(self.binary_mask[side_name][:, :, :] == 255, [0,0,0],  np.array(self.image[side_name]))
                # plt.imshow(overlay_image)
                # plt.show()
                self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8'))
                self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
                self.seg_canvas.config(width=self.image[side_name].width, height=self.image[side_name].height)
                self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)
        # self.after(0, self.loading.close)  # Close popup safely from main thread
        # self.page_ctrl.shared_data["segment_mask"] = self.binary_mask



    def set_canvas(self,side_select):
        self.input_canvas.config(width=self.image[side_select].width, height=self.image[side_select].height)
        # self.seg_canvas.config(width=self.image.width, height=self.image.height)

        if self.input_canvas_image:
            self.input_canvas.delete(self.input_canvas_image)
        self.input_canvas_image = self.input_canvas.create_image(0, 0, anchor="nw", image=self.tk_image[side_select])
        # self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.mask_canvas = Image.new("1", self.image[side_select].size, 0)
        self.mask_canvas_draw = ImageDraw.Draw(self.mask_canvas)
        self.seg_mask_canvas = Image.new("1", self.image[side_select].size, 0)
        self.seg_mask_canvas_draw = ImageDraw.Draw(self.seg_mask_canvas)

        self.side_label_key.set(side_select)
        self.side_label_show.set(self.show_side_name[side_select])

        if(self.binary_mask[side_select] is not None):
            # overlay_image = ((self.binary_mask[side_select]) * np.array(self.image[side_select])) + np.array(self.image[side_select])
            overlay_image = np.where(self.binary_mask[side_select][:, :, :] == 255, [0,0,0],  np.array(self.image[side_select]))
            self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8'))
            self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
        else:
            # self.seg_pil_image = Image.fromarray(self.image[side_select].astype('uint8'))
            self.seg_tk_img = ImageTk.PhotoImage(self.image[side_select])
        self.seg_canvas.config(width=self.image[side_select].width, height=self.image[side_select].height)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)

    def load_mask(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path:
            return
        self.binary_mask[self.side_label_key.get()] = preprocess_img(file_path,(self.config_data['img_top_h'], self.config_data['img_top_w'])) # open file by pillow
        self.binary_mask[self.side_label_key.get()] = binary_change(self.binary_mask[self.side_label_key.get()])
        # self.binary_mask[self.side_label_key.get()] = crop_center(self.binary_mask[self.side_label_key.get()],(self.config_data['model_input'][0],self.config_data['model_input'][1])) # crop (downscale) to match model input size
        self.binary_mask[self.side_label_key.get()] = up_downscale(self.binary_mask[self.side_label_key.get()],(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
        # overlay_image = ((self.binary_mask[self.side_label_key.get()]) * np.array(self.image[self.side_label_key.get()])) + np.array(self.image[self.side_label_key.get()])
        overlay_image = np.where(self.binary_mask[self.side_label_key.get()][:, :, :] == 255, [0,0,0],  np.array(self.image[self.side_label_key.get()]))
        # plt.imshow(overlay_image)
        # plt.show()
        self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8'))
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
        self.seg_canvas.config(width=self.image[self.side_label_key.get()].width, height=self.image[self.side_label_key.get()].height)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)

    def paint(self, event):
        # if not self.mask_canvas_draw:
        #     return

        x, y = event.x, event.y
        self.brush_size = 5
        self.seg_canvas.create_oval(x - self.brush_size, y - self.brush_size,
                                x + self.brush_size, y + self.brush_size,
                                fill="black", outline="black")
        self.seg_mask_canvas_draw.ellipse([x - self.brush_size, y - self.brush_size,
                                x + self.brush_size, y + self.brush_size], fill=1)
        combined_mask = np.maximum(self.binary_mask[self.side_label_key.get()], np.array(self.seg_mask_canvas.convert('RGB')))
        self.binary_mask[self.side_label_key.get()] = combined_mask

        
    def draw_rec_press(self,event):
        # Delete previous rectangle if exists
        if self.rect:
            self.input_canvas.delete(self.rect)
            self.rect = None
        self.start_x, self.start_y = event.x, event.y
        self.rect = self.input_canvas.create_rectangle(self.start_x,self.start_y,self.start_x,self.start_y,
                                outline="red")
        
    def draw_rec_move(self,event):
        if self.rect:
            self.input_canvas.coords(self.rect,self.start_x,self.start_y,event.x,event.y)
        
    def draw_rec_release(self,event):
        self.end_x = event.x
        self.end_y = event.y
        # print(f"draw rectangle xyxy format {self.start_x}, {self.start_y}, {event.x}, {event.y}")
        self.mask_canvas_draw.rectangle([self.start_x, self.start_y,
                                self.end_x, self.end_y], fill=0)
    
    def clear_rec(self,event):
        if self.rect:
            self.input_canvas.delete(self.rect)
            self.rect = None
            self.start_x,self.start_y,self.end_x,self.end_y = None, None, None, None
            # print('clear rectangle')
        
    def direct_draw(self):
        mask = np.full((self.image[self.side_label_key.get()].width,self.image[self.side_label_key.get()].height,3), 0, np.uint8) ## black background
        color = (255, 255, 255)
        thickness = -1
        self.binary_mask[self.side_label_key.get()] = cv2.rectangle(mask, (self.start_x, self.start_y), (self.end_x, self.end_y), color, thickness)
        # overlay_image = ((self.binary_mask[self.side_label_key.get()]) * np.array(self.image[self.side_label_key.get()])) + np.array(self.image[self.side_label_key.get()])
        overlay_image = np.where(self.binary_mask[self.side_label_key.get()][:, :, :] == 255, [0,0,0],  np.array(self.image[self.side_label_key.get()]))
        # plt.imshow(overlay_image)
        # plt.show()
        self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8'))
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
        self.seg_canvas.config(width=self.image[self.side_label_key.get()].width, height=self.image[self.side_label_key.get()].height)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)

    def clear_mask(self):
        self.binary_mask[self.side_label_key.get()] = None
        # self.binary_mask[self.side_label_key.get()] = binary_change(self.binary_mask[self.side_label_key.get()])
        # self.binary_mask[self.side_label_key.get()] = crop_center(self.binary_mask[self.side_label_key.get()],(self.config_data['model_input'][0],self.config_data['model_input'][1])) # crop (downscale) to match model input size
        # self.binary_mask[self.side_label_key.get()] = up_downscale(self.binary_mask[self.side_label_key.get()],(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
        # overlay_image = ((self.blank_mask) * np.array(self.image[self.side_label_key.get()])) + np.array(self.image[self.side_label_key.get()])
        overlay_image = np.where(self.blank_mask[:, :, :] == 255, [0,0,0],  np.array(self.image[self.side_label_key.get()]))
        # plt.imshow(overlay_image)
        # plt.show()
        self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8'))
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
        self.seg_canvas.config(width=self.image[self.side_label_key.get()].width, height=self.image[self.side_label_key.get()].height)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)

    def get_binary_mask(self):
        if self.mask:
            self.mask = (self.mask).convert('RGB')
            return np.array(self.mask, dtype=np.uint8)
        return None
    
    def new_config(self):
        new_config_name = simpledialog.askstring("Input", "กรอกชื่อ config file ใหม่")
        # project_root = os.path.abspath(os.path.join(current_file, "..", ".."))  # Adjust depth accordingly
        save_path = './mask/'
        # print(save_path)
        for count, side_name in enumerate(self.binary_mask):
            if(self.binary_mask[side_name] is not None):
                new_img_path = save_path+new_config_name+"_"+side_name+".jpg"
                tmp_save = up_downscale(self.binary_mask[side_name],(self.config_data['img_top_w'], self.config_data['img_top_h']),'upscale',self.page_ctrl.shared_data['type_process_scale']) # resize image to match real input size
                tmp_save = Image.fromarray(tmp_save.astype('uint8'))
                tmp_save.save(new_img_path)
                self.config_data["mask"][side_name] = new_img_path
                self.config_data["inpaint_side"][side_name] = True
        # edit config
        edit_json_by_key('config_gan_ui.json',new_config_name,self.config_data,'add')
    
    def save_config(self):
        # save image
        save_path = './mask/'
        config_name = self.page_ctrl.shared_data["config_name"]
        for count, side_name in enumerate(self.binary_mask):
            if(self.binary_mask[side_name] is not None):
                new_img_path = save_path+config_name+"_"+side_name+".jpg"
                tmp_save = up_downscale(self.binary_mask[side_name],(self.config_data['img_top_w'], self.config_data['img_top_h']),'upscale',self.page_ctrl.shared_data['type_process_scale']) # resize image to match real input size
                tmp_save = Image.fromarray(tmp_save.astype('uint8'))
                tmp_save.save(new_img_path)
                self.config_data["mask"][side_name] = new_img_path
                self.config_data["inpaint_side"][side_name] = True
        # edit config
        edit_json_by_key('config_gan_ui.json',config_name,self.config_data,'update')
        

    def save_mask(self):
        if self.binary_mask[self.side_label_key.get()] is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPG files", "*.jpg")])
            if save_path:
                # tmp_save = pad_images_to_same_size(self.binary_mask[self.side_label_key.get()])
                tmp_save = up_downscale(self.binary_mask[self.side_label_key.get()],(self.config_data['img_top_w'], self.config_data['img_top_h']),'upscale',self.page_ctrl.shared_data['type_process_scale']) # resize image to match real input size
                tmp_save = Image.fromarray(tmp_save.astype('uint8'))
                tmp_save.save(save_path)
                print("Mask saved to:", save_path)
                messagebox.showinfo("Info", f"Mask saved to: {save_path}")

    def reload_config(self):
        self.pre_process_mask()
        self.set_canvas(self.side_label_key.get())

    def start_thread_run_segment(self):
        self.loading = LoadingPopup(self, "Processing...")
        threading.Thread(target=self.run_segment).start()

    def run_segment(self):
        # self.seg_progress.start()
        self.binary_mask[self.side_label_key.get()] = bbox("per",np.array(self.image[self.side_label_key.get()]), [self.start_x,self.start_y,self.end_x,self.end_y], self.page_ctrl.shared_data['segment_model'])
        self.binary_mask[self.side_label_key.get()] = (self.binary_mask[self.side_label_key.get()].squeeze()).astype(np.uint8)
        self.binary_mask[self.side_label_key.get()] = cv2.cvtColor(self.binary_mask[self.side_label_key.get()], cv2.COLOR_GRAY2BGR)
        self.binary_mask[self.side_label_key.get()] = self.resize_mask(self.binary_mask[self.side_label_key.get()])
        # tmp_image = self.image[:]
        # overlay_image = ((self.binary_mask[self.side_label_key.get()]) * np.array(self.image[self.side_label_key.get()])) + np.array(self.image[self.side_label_key.get()])
        overlay_image = np.where(self.binary_mask[self.side_label_key.get()][:, :, :] == 255, [0,0,0],  np.array(self.image[self.side_label_key.get()]))
        # plt.imshow(overlay_image)
        # plt.show()
        self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8'))
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
        self.seg_canvas.config(width=self.image[self.side_label_key.get()].width, height=self.image[self.side_label_key.get()].height)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)
        # pil_image = Image.fromarray(per_img.astype('uint8'))
        # self.image = pil_image
        # ## --------------------------------
        # self.tk_image = ImageTk.PhotoImage(self.image)
        # self.seg_progress.stop()
        # self.seg_progress.pack_forget()
        self.after(0, self.loading.close)

    def resize_mask(self,mask, scale=11):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale)) 
        dilated_mask = cv2.dilate(mask, kernel)
        # h, w = mask.shape[:2]
        # new_size = (int(w * scale), int(h * scale))
        # resized = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        # resized = crop_center(resized,(self.config_data['model_input'][0],self.config_data['model_input'][1]))
        return dilated_mask

    def next_step(self):
        # self.page_ctrl.shared_data["input_text"].set("Hello from segment page")
        self.page_ctrl.shared_data["segment_mask"] = self.binary_mask
        self.page_ctrl.shared_data["mask_edited"] = True
        self.page_ctrl.show_frame('ImageProcessingApp')
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

    def back_step(self):
        self.page_ctrl.show_frame('ImageProcessingApp')


    def build_left(self):
        # print(self.page_ctrl.shared_data['input_img'])
        # if(self.page_ctrl.shared_data['input_img'] != None):
        #     self.load_image()
        label = tk.Label(self.left_frame, text="Input", font=("Arial", 14), justify="center", fg="green", bg="white")
        label.pack(anchor="c", padx=10, pady=10)

        # Placeholder for input image
        # self.input_canvas = tk.Label(self.left_frame, text="[Input Image]", bg="lightblue", width=40, height=20)
        self.input_canvas.pack(padx=10, pady=10)

    def build_toolBar(self):
        # Adding File Menu and commands
        # Fake menu with buttons
        btn_file = Menubutton(self, text="File", relief="raised")
        file_menu = Menu(btn_file, tearoff=0)
        file_menu.add_command(label="Open", command=lambda: print("Open in frame"))
        file_menu.add_command(label="Exit", command=self.parent_frame.quit)
        btn_file.config(menu=file_menu)
        btn_file.pack(side="left", padx=5)

    def build_center(self):
        buttons = ["Load new image", "Load config", "Rectangle Mask", "Segment", "Save imge", "Save config", "Clear", "Next step"]
        bt_action = [lambda: self.load_mask(), lambda:self.reload_config(), lambda: self.direct_draw() ,lambda: self.run_segment(),lambda: self.save_mask(), lambda: self.save_config() ,lambda: self.clear_mask(), lambda: self.next_step()] # using lambda: ..., delay execution — the code inside runs only when the button is clicked. 
        for text,cmd in zip(buttons,bt_action):
            btn = tk.Button(self.center_frame, text=text,command=cmd, width=15)
            btn.pack(pady=10)

    def build_right(self):
        label = tk.Label(self.right_frame, text="Mask", font=("Arial", 14) , justify="center", fg="green", bg="white")
        label.pack(anchor="c", padx=10, pady=10)

        # Placeholder for mask image
        # self.mask_canvas = tk.Label(self.right_frame, text="[Mask Image]", bg="lightcoral", width=40, height=20)
        # self.mask_canvas.pack(padx=10, pady=10)
        # self.seg_progress.pack()
        self.seg_canvas.pack(padx=10, pady=10)

    def build_header(self):
        # Configure 3 columns in grid
        self.header_frame.grid_columnconfigure(0, weight=1)
        self.header_frame.grid_columnconfigure(1, weight=1)
        self.header_frame.grid_columnconfigure(2, weight=1)
        self.btn_left = tk.Button(self.header_frame, text='<',command=lambda: self.previous_image(), width=15, bg="green", fg="white")
        self.btn_left.grid(row=0, column=0, sticky="w")
        side_label = tk.Label(self.header_frame, textvariable=self.side_label_show, font=("Arial", 14), bg="green", fg="white")
        side_label.grid(row=0, column=1)
        self.btn_right = tk.Button(self.header_frame, text='>',command=lambda: self.next_image(), width=15, bg="green", fg="white")
        self.btn_right.grid(row=0, column=2, sticky="e")


