import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
import sys

# from .main_page import ImageProcessingApp

sys.path.append("..")
from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from tools.process_pano import Equirectangular,Perspective,inpaint_pano
from tools.utils import up_downscale, DictNavigator, binary_change
from tools.py360convert.py360convert import c2e, e2c
from tools.sam_tools.classification_torch import bbox
# --------------------- segment page ------------------------

class MaskApp(tk.Frame):
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
        # self.binary_mask[self.side_label_show.get()] = crop_center(self.binary_mask[self.side_label_show.get()],(self.config_data['model_input'][0],self.config_data['model_input'][1])) # crop (downscale) to match model input size
        self.binary_mask[self.side_label_show.get()] = up_downscale(self.binary_mask[self.side_label_show.get()],(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
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
                # tmp_save = pad_images_to_same_size(self.binary_mask[self.side_label_show.get()])
                tmp_save = up_downscale(self.binary_mask[self.side_label_show.get()],(self.config_data['img_top_w'], self.config_data['img_top_h']),'upscale',self.page_ctrl.shared_data['type_process_scale']) # resize image to match real input size
                tmp_save = Image.fromarray(tmp_save.astype('uint8'))
                tmp_save.save(save_path)
                print("Mask saved to:", save_path)

    def run_segment(self):
        self.binary_mask[self.side_label_show.get()] = bbox("per",np.array(self.image[self.side_label_show.get()]), [self.start_x,self.start_y,self.end_x,self.end_y], self.page_ctrl.shared_data['segment_model'])
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


