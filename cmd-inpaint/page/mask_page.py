import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from tkinter import filedialog, messagebox, ttk, Menu, Menubutton, simpledialog
import numpy as np
import cv2
import sys
import json
import threading
import os

from .loadingPopup_page import LoadingPopup
from . import find_file_path

sys.path.append("..")
from tools.process_img import preprocess_img
from tools.process_pano import Perspective
from tools.utils import up_downscale, DictNavigator, binary_change
from tools.py360convert.py360convert import  e2c
from tools.sam_tools.classification_torch import bbox
# --------------------- segment page ------------------------

def edit_json_by_key(file_path:str, key_to_match:str, new_value:dict,type_command:str):
    """
    Edits a JSON file by updating or adding a key-value pair.
    Args:
        file_path (str): The path to the JSON file.
        key_to_match (str): The key to update or add in the JSON file.
        new_value (dict): The new value to set for the specified key.
        type_command (str): The operation type, either 'update' to modify an existing key or 'add' to insert a new key.
    Raises:
        Shows a message box error if the file is not found or contains invalid JSON.
        Shows a message box error if attempting to add a key that already exists.
        Shows a message box info on successful update or addition.
    Note:
        This function uses `find_file_path` to resolve the file path and `messagebox` for user notifications.
    """
    
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
    # ----------- case: update current config ------------------------------
    if(type_command == 'update'):
        for key, value in data.items():
            if key == key_to_match: # search match config in json file
                data[key] = new_value # update value
                # edit_flag = True
                with open(file_path, 'w') as file: # write to json file
                    json.dump(data, file, indent=4) # Optional: indent for pretty formatting
                messagebox.showinfo("Info", "Config file updated successfully.") # show message box
    # ----------- case: add new config -------------------------------------
    elif(type_command == 'add'):
        currentKey_list = list(data.keys()) # get current config name
        if(key_to_match not in currentKey_list): # if not find new config name in current config list do:
            data[key_to_match] = new_value # add new value
            with open(file_path, 'w') as file: # write to json
                json.dump(data, file, indent=4) # Optional: indent for pretty formatting
            messagebox.showinfo("Info", "Add new config file successfully.") # show message box
        else:
            messagebox.showerror("Error", f"Already have config file name: {key_to_match}") # show error box if new name match older name
            return

class MaskApp(tk.Frame):
    """
    MaskApp is a Tkinter Frame-based application for interactive image segmentation and mask editing.
    This class provides a GUI for loading, displaying, and editing segmentation masks on multiple views of an image (e.g., top, front, right, etc.). It supports drawing rectangles, freehand painting, loading/saving masks, and running segmentation models. The application manages multiple image views, allows navigation between them, and synchronizes mask data with a shared configuration.
    Attributes:
        page_ctrl: Reference to the main controller for shared data and navigation to another frame(page).
        config_data: Configuration dictionary shared between pages.
        show_side_name: Mapping of internal side keys to display names.
        image: Dictionary of PIL images for each view.
        tk_image: Dictionary of Tkinter-compatible images for each view.
        image_pano: The input panorama image.
        image_pano_cube: List of cube map images derived from the panorama.
        rect: Current rectangle selection on the input canvas.
        mask_canvas, mask_canvas_draw: Canvas and drawing context for input mask.
        seg_mask_canvas, seg_mask_canvas_draw: Canvas and drawing context for segmentation mask.
        binary_mask: Dictionary of binary masks for each view.
        blank_mask: Blank mask template for clearing masks.
        start_x, start_y, end_x, end_y: Coordinates for rectangle selection.
        header_frame, left_frame, right_frame: GUI frames for header, input, and mask sections.
        input_canvas, seg_canvas: Canvases for displaying input and mask images.
        input_canvas_image, seg_canvas_image: Canvas image handles.
        side_label_key, side_label_show: Tkinter StringVars for current side selection.
        nav_image: Navigator object for cycling through image views.
        btn_left, btn_right: Navigation buttons.
    Methods:
        start_thread_on_show(): Start loading and preprocessing images in a separate thread.
        on_show(): Preprocess images and masks when the page is shown.
        pre_process_img(): Load and preprocess input images and cube maps.
        pre_process_mask(): Load and preprocess mask images for each view.
        set_canvas(side_select): Set the displayed image and mask for the selected view.
        load_mask(): Load a mask image from file for the current view.
        paint(event): Freehand paint on the mask canvas.
        draw_rec_press(event), draw_rec_move(event), draw_rec_release(event): Handle rectangle drawing events.
        clear_rec(event): Clear the current rectangle selection.
        direct_draw(): Draw a rectangle mask directly on the binary mask.
        clear_mask(): Clear the mask for the current view.
        get_binary_mask(): Retrieve the current binary mask as a NumPy array.
        new_config(): Save current masks as a new configuration.
        save_config(): Save current masks to the existing configuration.
        save_mask(): Save the current mask to a file.
        reload_config(): Reload masks from the configuration.
        start_thread_run_segment(): Start segmentation in a separate thread.
        run_segment(): Run the segmentation model on the selected region.
        resize_mask(mask, scale): Dilate the mask using a morphological operation.
        next_step(): Proceed to the next application step.
        next_image(), previous_image(): Navigate between image views.
        back_step(): Return to the previous application step.
        build_left(), build_center(), build_right(), build_header(): Build GUI components for the application.
    Usage:
        Instantiate MaskApp as part of a Tkinter application, providing a parent widget and a controller with shared data.
    """

    def __init__(self, parent, controller):
        super().__init__(parent, bd=0.5, relief="solid", highlightbackground="green", highlightcolor="green", highlightthickness=0.5)
        self.page_ctrl = controller # main container (from open page)
        # ------------ variable for segment task
        self.config_data = self.page_ctrl.shared_data["config_data"] # data shared between page
        self.show_side_name = { # name each side to show on header
                "top_view": "Bird view",
                "cube_front": "Front view",
                "cube_right": "Right view",
                "cube_back": "Back view",
                "cube_left": "Left view",
                "cube_top": "Upper view",
                "cube_bottom": "Bottom view"
            }
        self.image = { # stored pillow image (input) 
                "top_view": None,
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.tk_image = { # stored tkinter image
                "top_view": None,
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.image_pano = None # stored input panorama
        self.image_pano_cube = None # stored cube map list
        self.rect = None # rectangle paint from input section
        # Create a mask that you can draw on --------------------------
        self.mask_canvas = None # Input section: blank canvas (black px (0))
        self.mask_canvas_draw = None # Input section: area to draw into
        self.seg_mask_canvas = None # Mask section : blank canvas (black px (0))
        self.seg_mask_canvas_draw = None # Mask section: area to draw into
        self.binary_mask = { # stored numpy binary mask
                "top_view": None,
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        blank_mask = np.full((self.config_data['img_top_h'],self.config_data['img_top_w'],3), 0.0, np.float32) # blank mask (for clear mask) - black px (0)
        self.blank_mask = up_downscale(blank_mask,(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
        self.start_x,self.start_y,self.end_x,self.end_y = None, None, None, None # xyxy box
        # ----------- create element in header frame (show side name with action select side) --------------------------
        self.header_frame = tk.Frame(self, width=600, bg="green", height=50) # header (show 'side name' and 'left(previous) and right(next) button' to select side name)
        self.header_frame.pack(fill="both")
        self.side_label_key = tk.StringVar() # stored key name of side name for select in dict
        self.side_label_show = tk.StringVar() # stored name of current show side view
        self.nav_image = None # stored obj from navigation class (to left(previous) and right(next) button)
        self.btn_left = None # left(previous) button
        self.btn_right = None # right(next) button
        # ----------- create element in left frame (Input section) --------------------------
        self.left_frame = tk.Frame(self, width=300, bg="white", bd=0.5, relief="solid", highlightbackground="green", highlightcolor="green", highlightthickness=0.5)
        self.input_canvas = tk.Canvas(self.left_frame)
        self.input_canvas_image = None
        self.left_frame.pack(side="left", fill="both", expand=False)
        # ----------- create element in left frame (Mask section) --------------------------
        self.right_frame = tk.Frame(self, width=300, bg="white" , bd=0.5, relief="solid", highlightbackground="green", highlightcolor="green", highlightthickness=0.5)
        self.right_frame.pack(side="left", fill="both", expand=False)
        self.seg_canvas = tk.Canvas(self.right_frame)
        self.seg_canvas_image = None
        # ------------- Build section in Frame ----------------
        self.build_left()
        # self.build_toolBar()
        # self.build_center()
        self.build_right()
        self.build_header()
        # ------------ bind event to button --------------------------
        # Input section ------------------------------------
        self.input_canvas.bind("<ButtonPress-1>", self.draw_rec_press) # action: press left mouse to start draw rectangle, remove previous rectangle if exist
        self.input_canvas.bind("<B1-Motion>", self.draw_rec_move) # action: move mouse while still press left mouse to increase or decrease size of rectangle
        self.input_canvas.bind("<ButtonRelease-1>", self.draw_rec_release) # action: release left mouse to draw rectangle in final size
        self.input_canvas.bind("<Button-3>",self.clear_rec) # action: press right mouse to delete rectangle if exist
        # Mask section -------------------------------------
        self.seg_canvas.bind("<B1-Motion>", self.paint) # action: move left mouse to draw (brush) mask

    def start_thread_on_show(self):
        """
        Starts a new thread to execute the `on_show` method and displays a loading popup.
        This method creates and shows a `LoadingPopup` to indicate processing,
        then starts the `on_show` method in a separate thread to avoid blocking the UI.
        Notes:
            - will start on landing mask page
        """
        self.loading = LoadingPopup(self,"Processing...") # call loading popup
        threading.Thread(target = self.on_show).start() # start thread to prepare data to show on frame

    def on_show(self):
        """
        Handles the logic to be executed when the page is shown.
        - Loads configuration data from shared data.
        - Pre-processes the image for display or further operations.
        - If the mask has not been edited, resets the binary mask and pre-processes the mask.
        - Sets the canvas to display the first image.
        - Closes the loading popup safely from the main thread.
        """
        self.config_data = self.page_ctrl.shared_data["config_data"] # get selected config data from shared data
        self.pre_process_img() # process image for show
        if(not self.page_ctrl.shared_data["mask_edited"]): # if mask is not edited process mask from config file
            reset_segment_mask = {key: None for key in self.binary_mask}
            self.binary_mask = reset_segment_mask
            self.pre_process_mask()
        self.set_canvas(list(self.image.keys())[0]) # set first element of image dict to show
        self.after(0, self.loading.close) # when finish process close loading popup

    def pre_process_img(self):
        """
        Pre-processes the input panorama image to generate top-view (bird view) and cube-map images,
        resizes them to match the model input size, and stores them as both PIL and Tkinter images.
        Steps performed:
            1. Loads and preprocesses the input panorama image.
            2. Generates a top-view (bird view) perspective image from the panorama.
            3. Resizes the top-view image to the model's input size and stores it as a PIL and Tkinter image.
            4. Converts the panorama image to a cube-map (six faces).
            5. For each cube face (excluding the top view), resizes and stores as PIL and Tkinter images.
            6. Initializes a DictNavigator with the processed images for navigation.
        Modifies:
            self.image: Dictionary containing PIL images for top view and cube faces.
            self.tk_image: Dictionary containing Tkinter PhotoImage objects for top view and cube faces.
            self.nav_image: DictNavigator object for image navigation.
            self.image_pano: Preprocessed panorama image.
            self.image_pano_cube: List of cube-map face images.
            self.full_top_img: Numpy array of the full top-view image.
        Requires:
            - self.page_ctrl.shared_data['input_img']: Input panorama image (path to image).
            - self.config_data: Configuration dictionary with image sizes and coordinates.
            - preprocess_img, Perspective, up_downscale, e2c, DictNavigator: External functions/classes.
        """

        # ----------- 1. panorama -> top-view -----------------------------
        self.image_pano = preprocess_img(self.page_ctrl.shared_data['input_img'],img_size=(self.config_data['img_pano_w'],self.config_data['img_pano_h'])) # read image
        ## select top view ---------------##
        equ = Perspective(self.image_pano)    # Load equirectangular image to class
        self.full_top_img = equ.GetPerspective(167, self.config_data['coor'], -90, self.config_data['img_top_h'], self.config_data['img_top_w'])  # Specify parameters(FOV, theta, phi, height, width) to get perspective (bird view)
        # crop (downscale) to match model input size | resize (downscale) to match model input size
        per_img = up_downscale(self.full_top_img,(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
        # Convert to PIL Image
        pil_image = Image.fromarray(per_img.astype('uint8'))
        self.image["top_view"] = pil_image # stored pillow image (bird view)
        self.tk_image["top_view"] = ImageTk.PhotoImage(self.image["top_view"]) # stored tkinter image (bird view)
        # ------------------- 2. panorama -> cube-map -------------------------------
        self.image_pano_cube = e2c(self.image_pano, face_w=1000, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        for count, side_name in enumerate(self.image): # run through all side map
            if(count>0): # except first element (bird view/ top view)
                count = count - 1
                # crop (downscale) to match model input size | resize (downscale) to match model input size
                car_cube_img = up_downscale(self.image_pano_cube[count],(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
                self.image[side_name] = Image.fromarray(car_cube_img.astype('uint8')) # stored pillow image (cube map)
                self.tk_image[side_name] = ImageTk.PhotoImage(self.image[side_name]) # stored tkinter image (cube map)
        self.nav_image = DictNavigator(self.image) # add dict image data to obj

    def pre_process_mask(self):
        """
        Pre-processes the binary mask images for each side defined in the configuration.
        For each side in the binary mask dictionary:
            - Loads the mask image from the file path specified in the configuration.
            - Preprocesses the image to the specified top height and width.
            - Converts the image to a binary mask.
            - Downscales the mask to match the model's input size using the specified scaling method.
            - Overlays the binary mask on the corresponding input image, replacing masked (white) pixels with black.
            - Converts the overlay image to a PIL Image and then to a Tkinter PhotoImage for display.
            - Configures the canvas size and displays the processed image for further drawing or editing.
        Assumes that supporting functions (`preprocess_img`, `find_file_path`, `binary_change`, `up_downscale`) and required attributes
        (`self.config_data`, `self.binary_mask`, `self.image`, `self.page_ctrl.shared_data`, `self.seg_canvas`) are properly defined.
        """
        for count, side_name in enumerate(self.binary_mask): # run through binary mask dict
            file_path = self.config_data["mask"][side_name] # get file path to mask each side
            if(file_path != ""): # if have mask
                self.binary_mask[side_name] = preprocess_img(find_file_path(file_path),(self.config_data['img_top_h'], self.config_data['img_top_w'])) # read file
                self.binary_mask[side_name] = binary_change(self.binary_mask[side_name]) # binarized
                # crop (downscale) to match model input size | resize (downscale) to match model input size
                self.binary_mask[side_name] = up_downscale(self.binary_mask[side_name],(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
                # overlay_image = ((self.binary_mask[side_name]) * np.array(self.image[side_name])) + np.array(self.image[side_name])
                # overlay mask on view (replace where mask have white px with black color(0) on input image)
                overlay_image = np.where(self.binary_mask[side_name][:, :, :] == 255, [0,0,0],  np.array(self.image[side_name]))
                self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8')) # stored pillow image
                self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image) # stored tkinter image
                self.seg_canvas.config(width=self.image[side_name].width, height=self.image[side_name].height) # set blank mask for draw
                self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img) # allow to draw shape on self.seg_canvas



    def set_canvas(self,side_select:str):
        """
        Sets up the input and segmentation canvases for the selected image side.
        This method performs the following actions:
        - Configures the input canvas size to match the selected image.
        - Displays the selected image on the input canvas.
        - Initializes black-and-white mask canvases for drawing and segmentation.
        - Updates labels to reflect the currently selected side.
        - If a binary mask exists for the selected side, overlays the mask on the image and displays it on the segmentation canvas.
        - Otherwise, displays the original image on the segmentation canvas.
        Args:
            side_select (str): The key indicating which side/image to display and set up.
        """
        # --------------- set area for input section --------------------
        self.input_canvas.config(width=self.image[side_select].width, height=self.image[side_select].height) # set size of canvas to show image
        if self.input_canvas_image:
            self.input_canvas.delete(self.input_canvas_image) # if have previous image delete first
        self.input_canvas_image = self.input_canvas.create_image(0, 0, anchor="nw", image=self.tk_image[side_select]) # set showed image by the side that user select (current side view)
        # ------------------ draw mask area setting ------------------------------
        self.mask_canvas = Image.new("1", self.image[side_select].size, 0) # Input section: create black and white (fill with black px (0))
        self.mask_canvas_draw = ImageDraw.Draw(self.mask_canvas) # Input section: allow to draw shape on self.mask_canvas
        self.seg_mask_canvas = Image.new("1", self.image[side_select].size, 0) # Mask section: create black and white (fill with black px (0))
        self.seg_mask_canvas_draw = ImageDraw.Draw(self.seg_mask_canvas) # Mask section: allow to draw shape on self.mask_canvas
        # ---------------------------------------------------------------------------
        self.side_label_key.set(side_select) # set key of selected side
        self.side_label_show.set(self.show_side_name[side_select]) # set text for show on header
        if(self.binary_mask[side_select] is not None): # if have mask
            # overlay_image = ((self.binary_mask[side_select]) * np.array(self.image[side_select])) + np.array(self.image[side_select])
            overlay_image = np.where(self.binary_mask[side_select][:, :, :] == 255, [0,0,0],  np.array(self.image[side_select])) # overlay by replace area with white(255) on mask with white (0) on image
            self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8')) # stored pillow
            self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image) # stored tkinter image
        else: # if not have mask show only input image
            self.seg_tk_img = ImageTk.PhotoImage(self.image[side_select]) 
        self.seg_canvas.config(width=self.image[side_select].width, height=self.image[side_select].height) # set size of canvas to show segment mask
        if self.seg_canvas_image:
            self.seg_canvas.delete(self.seg_canvas_image)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img) # show overlay image on canvas

    def load_mask(self):
        """
        Loads a mask image selected by the user, preprocesses and binarizes it, resizes it to match the model input size,
        overlays the mask on the corresponding image for visualization, and updates the display canvas.
        Steps performed:
            1. Prompts the user to select an image file for the mask.
            2. Preprocesses the selected image to the required dimensions.
            3. Binarizes the preprocessed mask image.
            4. Resizes (downscales) the mask to match the model's input size.
            5. Overlays the binary mask onto the original image to highlight the selected area.
            6. Converts the overlay image to a format suitable for Tkinter display.
            7. Updates the canvas widget to display the overlay image.
        Returns:
            None
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]) # show pop up folder for user to select new mask
        if not file_path: # if not selected don't do anything
            return
        self.binary_mask[self.side_label_key.get()] = preprocess_img(file_path,(self.config_data['img_top_h'], self.config_data['img_top_w'])) # read image
        self.binary_mask[self.side_label_key.get()] = binary_change(self.binary_mask[self.side_label_key.get()]) # binarized
        # crop (downscale) to match model input size | resize (downscale) to match model input size
        self.binary_mask[self.side_label_key.get()] = up_downscale(self.binary_mask[self.side_label_key.get()],(self.config_data['model_input'][0],self.config_data['model_input'][1]),'downscale',self.page_ctrl.shared_data['type_process_scale']) # resize (downscale) to match model input size
        # overlay_image = ((self.binary_mask[self.side_label_key.get()]) * np.array(self.image[self.side_label_key.get()])) + np.array(self.image[self.side_label_key.get()])
        overlay_image = np.where(self.binary_mask[self.side_label_key.get()][:, :, :] == 255, [0,0,0],  np.array(self.image[self.side_label_key.get()])) # overlay image with mask to show select area
        self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8')) # stored pillow image
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image) # stored tkinter image
        self.seg_canvas.config(width=self.image[self.side_label_key.get()].width, height=self.image[self.side_label_key.get()].height) # set width height of canvas
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img) # show overlay image

    def paint(self, event):
        """
        Handles the paint event on the segmentation canvas.
        Draws a filled oval (representing a brush stroke) at the cursor position on both the visible canvas and the mask canvas.
        Updates the binary mask for the current side label by combining the existing mask with the new brush stroke.
        Args:
            event: An event object containing the x and y coordinates of the cursor.
        """
        x, y = event.x, event.y
        self.brush_size = 5
        self.seg_canvas.create_oval(x - self.brush_size, y - self.brush_size,
                                x + self.brush_size, y + self.brush_size,
                                fill="black", outline="black") # size of brush
        self.seg_mask_canvas_draw.ellipse([x - self.brush_size, y - self.brush_size,
                                x + self.brush_size, y + self.brush_size], fill=1) # draw brush by path of moving mouse
        combined_mask = np.maximum(self.binary_mask[self.side_label_key.get()], np.array(self.seg_mask_canvas.convert('RGB'))) # combined previous mask with brush
        self.binary_mask[self.side_label_key.get()] = combined_mask # show combined mask on canvas

        
    def draw_rec_press(self,event):
        """
        Handles the mouse press event to start drawing a rectangle on the input canvas.
        If a rectangle already exists, it is deleted before starting a new one.
        Initial coordinates are set based on the mouse event position, and a new rectangle
        is created with its outline in red.
        Args:
            event: The Tkinter event object containing the x and y coordinates of the Left mouse press.
        """
        # Delete previous rectangle if exists
        if self.rect:
            self.input_canvas.delete(self.rect)
            self.rect = None
        self.start_x, self.start_y = event.x, event.y
        self.rect = self.input_canvas.create_rectangle(self.start_x,self.start_y,self.start_x,self.start_y,
                                outline="red") # draw red rectangle
        
    def draw_rec_move(self,event):
        """
        Handles the mouse movement event to update the coordinates of a rectangle being drawn on the input canvas.
        Args:
            event: The event object containing the current mouse position (event.x, event.y).
        Updates:
            - Adjusts the coordinates of the existing rectangle (`self.rect`) on `self.input_canvas` to reflect the new mouse position as the user drags the mouse.
        """

        if self.rect:
            self.input_canvas.coords(self.rect,self.start_x,self.start_y,event.x,event.y)
        
    def draw_rec_release(self,event):
        """
        Handles the mouse release event to draw a edged rectangle on the mask canvas.
        When the mouse button is released, this method captures the ending coordinates
        (event.x, event.y) and draws a edged rectangle from the starting coordinates
        (self.start_x, self.start_y) to the ending coordinates on the mask canvas.
        Args:
            event: An event object containing the x and y coordinates of the mouse release.
        """
        self.end_x = event.x
        self.end_y = event.y
        # print(f"draw rectangle xyxy format {self.start_x}, {self.start_y}, {event.x}, {event.y}")
        self.mask_canvas_draw.rectangle([self.start_x, self.start_y,
                                self.end_x, self.end_y], fill=0)
    
    def clear_rec(self,event):
        """
        Clears the currently drawn rectangle from the input canvas, if it exists.
        This method deletes the rectangle shape from the canvas, resets the rectangle reference,
        and sets the rectangle's start and end coordinates to None.
        Typically used as an event handler to clear a selection rectangle.
        Args:
            event: The event object that triggered this method (Right mouse).
        """
        if self.rect:
            self.input_canvas.delete(self.rect)
            self.rect = None
            self.start_x,self.start_y,self.end_x,self.end_y = None, None, None, None
            # print('clear rectangle')
        
    def direct_draw(self):
        """
        Draws a rectangular mask on the current image, overlays the mask onto the image to highlight the selected area, 
        and updates the display canvas with the resulting image.
        The method performs the following steps:
        1. Creates a black mask of the same size as the current image.
        2. Draws a white-filled rectangle on the mask to represent the selected area.
        3. Applies the mask to the image, overlaying black on the detected object area.
        4. Converts the resulting image to a PIL Image and then to a Tkinter-compatible image.
        5. Updates the canvas size and displays the overlay image on the mask section.
        Attributes used:
            - self.image: Dictionary of images keyed by side label.
            - self.side_label_key: Function or property to get the current side label key.
            - self.start_x, self.start_y: Coordinates for the top-left corner of the rectangle.
            - self.end_x, self.end_y: Coordinates for the bottom-right corner of the rectangle.
            - self.binary_mask: Dictionary to store binary masks for each side label.
            - self.seg_pil_image: Stores the resulting PIL image.
            - self.seg_tk_img: Stores the Tkinter-compatible image.
            - self.seg_canvas: The Tkinter canvas for displaying the mask.
            - self.seg_canvas_image: Reference to the image displayed on the canvas.
        """
        mask = np.full((self.image[self.side_label_key.get()].width,self.image[self.side_label_key.get()].height,3), 0, np.uint8) ## black background
        color = (255, 255, 255) # white color as selected area
        thickness = -1
        self.binary_mask[self.side_label_key.get()] = cv2.rectangle(mask, (self.start_x, self.start_y), (self.end_x, self.end_y), color, thickness) # draw rectangle as binary mask
        # overlay_image = ((self.binary_mask[self.side_label_key.get()]) * np.array(self.image[self.side_label_key.get()])) + np.array(self.image[self.side_label_key.get()])
        overlay_image = np.where(self.binary_mask[self.side_label_key.get()][:, :, :] == 255, [0,0,0],  np.array(self.image[self.side_label_key.get()])) # overlay black on detect object area
        self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8')) # stored pillow image
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image) # stored tkinter image
        self.seg_canvas.config(width=self.image[self.side_label_key.get()].width, height=self.image[self.side_label_key.get()].height) # set width height of mask canvas
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img) # show overlay image on mask section

    def clear_mask(self):
        """
        Clears the current mask for the selected side label, resets the binary mask,
        overlays a blank mask over the image to hide the selected area, and updates
        the displayed image on the canvas.
        This method performs the following steps:
            1. Resets the binary mask for the current side label to None.
            2. Creates an overlay image by applying the blank mask to the original image,
               effectively hiding the selected area.
            3. Converts the overlay image to a PIL Image and then to a Tkinter PhotoImage.
            4. Updates the canvas dimensions and displays the new image.
        Assumes that the following attributes are defined:
            - self.binary_mask: Dictionary mapping side labels to their binary masks.
            - self.side_label_key: A key or method to get the current side label.
            - self.blank_mask: A mask image used to overlay on the original image.
            - self.image: Dictionary mapping side labels to their images.
            - self.seg_pil_image: PIL Image object for the segmented image.
            - self.seg_tk_img: Tkinter PhotoImage object for the segmented image.
            - self.seg_canvas: Tkinter Canvas widget for displaying the image.
            - self.seg_canvas_image: Canvas image object ID.
        """
        self.binary_mask[self.side_label_key.get()] = None # reset binary mask
        # overlay_image = ((self.blank_mask) * np.array(self.image[self.side_label_key.get()])) + np.array(self.image[self.side_label_key.get()])
        overlay_image = np.where(self.blank_mask[:, :, :] == 255, [0,0,0],  np.array(self.image[self.side_label_key.get()])) # overlay blank mask over image (not show selected area)
        self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8')) #stored pillow image
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image) # stored tkinter image
        self.seg_canvas.config(width=self.image[self.side_label_key.get()].width, height=self.image[self.side_label_key.get()].height) # set width and height of mask section
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img) # show overlay image on mask section

    def get_binary_mask(self):
        """
        Converts the current mask image to a NumPy array in RGB format.
        If a mask exists, it is converted to RGB mode and then to a NumPy array of type uint8.
        If no mask is present, returns None.
        Returns:
            np.ndarray or None: The mask image as a NumPy array in RGB format, or None if no mask exists.
        """
        if self.mask:
            self.mask = (self.mask).convert('RGB')
            return np.array(self.mask, dtype=np.uint8)
        return None
    
    def new_config(self):
        """
        Prompts the user to enter a new configuration name, saves the current binary masks for each side with the new name,
        updates the configuration data with the new mask paths and flags, and adds the new configuration to the JSON file.
        Steps:
        1. Asks the user for a new configuration file name using a dialog.
        2. Iterates over each side in the binary mask dictionary.
        3. For each side with a mask:
            - Constructs a new image path based on the config name and side name.
            - Upscales the mask to match the original input size.
            - Converts the mask to a PIL Image and saves it to the new path.
            - Updates the configuration data with the new mask path and sets the inpaint flag.
        4. Adds the updated configuration data to the 'config_gan_ui.json' file under the new config name.
        """
        new_config_name = simpledialog.askstring("Input", "กรอกชื่อ config file ใหม่") # ask user to input new config name
        save_path = './mask/' # save to mask folder
        for count, side_name in enumerate(self.binary_mask): # run through all side
            if(self.binary_mask[side_name] is not None): # if have mask on said side
                new_img_path = save_path+new_config_name+"_"+side_name+".jpg" # set new image name to reflect config name and side name 
                # padding (upscale) to match original input | resize (upscale) image to match real input size
                tmp_save = up_downscale(self.binary_mask[side_name],(self.config_data['img_top_w'], self.config_data['img_top_h']),'upscale',self.page_ctrl.shared_data['type_process_scale']) # resize (upscale) image to match real input size
                tmp_save = Image.fromarray(tmp_save.astype('uint8')) # cast to pillow
                tmp_save.save(new_img_path) # save new mask
                self.config_data["mask"][side_name] = new_img_path # set new mask path to current setting
                self.config_data["inpaint_side"][side_name] = True # set flag that said side have mask
        # add new config
        edit_json_by_key('config_gan_ui.json',new_config_name,self.config_data,'add')
    
    def save_config(self):
        """
        Saves the current mask configuration and associated images.
        This method performs the following actions:
        1. Iterates over each side in the binary mask dictionary.
        2. For each non-None mask, upscales the mask to match the original input size.
        3. Saves the upscaled mask image to the designated directory with a filename based on the config name and side.
        4. Updates the configuration data with the new mask image path and sets the inpaint flag for the side.
        5. Updates the configuration JSON file with the modified configuration data.
        Raises:
            KeyError: If required keys are missing in shared_data or config_data.
            Exception: If image saving or JSON editing fails.
        """
        # save image
        save_path = './mask/' # save path for mask
        config_name = self.page_ctrl.shared_data["config_name"] # get current config name
        for count, side_name in enumerate(self.binary_mask):
            if(self.binary_mask[side_name] is not None):
                new_img_path = save_path+config_name+"_"+side_name+".jpg"
                # padding (upscale) to match original input | resize (upscale) image to match real input size
                tmp_save = up_downscale(self.binary_mask[side_name],(self.config_data['img_top_w'], self.config_data['img_top_h']),'upscale',self.page_ctrl.shared_data['type_process_scale']) # resize image to match real input size
                tmp_save = Image.fromarray(tmp_save.astype('uint8'))
                tmp_save.save(new_img_path)
                self.config_data["mask"][side_name] = new_img_path
                self.config_data["inpaint_side"][side_name] = True
        # update config
        edit_json_by_key('config_gan_ui.json',config_name,self.config_data,'update')
        

    def save_mask(self):
        """
        Saves the current binary mask for the selected side to a JPG file.
        This method checks if a binary mask exists for the currently selected side. 
        If so, it prompts the user to choose a file path for saving the mask image using a file dialog. 
        The mask is upscaled to match the original input image size before saving. After saving, a message box notifies the user of the successful operation.
        Steps:
            1. Check if a binary mask exists for the selected side.
            2. Open a file dialog to select the save location and filename.
            3. Upscale the mask to match the original input size.
            4. Save the mask as a JPG image.
            5. Notify the user upon successful save.
        Raises:
            Exception: If there is an error during the saving process.
        Side Effects:
            - Opens file dialog and message box.
            - Writes a JPG file to disk.
            - Prints the save path to the console.
        """
        if self.binary_mask[self.side_label_key.get()] is not None: # if have mask in selected side
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPG files", "*.jpg")]) # ask name for mask image to save
            if save_path:
                # padding (upscale) to match original input | resize (upscale) image to match real input size
                tmp_save = up_downscale(self.binary_mask[self.side_label_key.get()],(self.config_data['img_top_w'], self.config_data['img_top_h']),'upscale',self.page_ctrl.shared_data['type_process_scale']) # resize image to match real input size
                tmp_save = Image.fromarray(tmp_save.astype('uint8'))
                tmp_save.save(save_path)
                print("Mask saved to:", save_path)
                messagebox.showinfo("Info", f"Mask saved to: {save_path}") # pop up message ox to notify saving success

    def reload_config(self):
        """
        Reloads the configuration for the mask page.
        This method performs any necessary preprocessing on the mask and reset mask in the canvas
        based on the currently selected side label key.
        """
        self.pre_process_mask() # reprocess mask by reading from config file
        self.set_canvas(self.side_label_key.get()) # show current page in frame

    def start_thread_run_segment(self):
        """
        Starts a new thread to run the segmentation process and displays a loading popup.
        This method creates and shows a LoadingPopup to indicate that processing is in progress,
        then starts a new thread to execute the `run_segment` method without blocking the main UI thread.
        """
        self.loading = LoadingPopup(self, "Processing...")
        threading.Thread(target=self.run_segment).start()

    def run_segment(self):
        """
        Runs the segmentation process on the selected region of the image using a segmentation model.
        This method performs the following steps:
        1. Applies the segmentation model to the user-selected bounding box area of the image to generate a binary mask.
        2. Processes the binary mask by squeezing, converting to uint8, and converting to BGR color format.
        3. Resizes the binary mask to match the image dimensions.
        4. Creates an overlay image by masking out the segmented area (setting it to black) and keeping the rest of the original image.
        5. Converts the overlay image to a PIL Image and then to a Tkinter-compatible PhotoImage.
        6. Updates the segmentation canvas with the new image.
        7. Closes the loading dialog asynchronously.
        Assumes the following instance attributes are available:
            - self.binary_mask: Dictionary to store binary masks for each side label.
            - self.side_label_key: Key to select the current side/image.
            - self.image: Dictionary of images for each side label.
            - self.page_ctrl.shared_data['segment_model']: The segmentation model to use.
            - self.resize_mask: Method to resize the mask.
            - self.seg_canvas: Tkinter canvas for displaying the segmentation result.
            - self.loading: Loading dialog to be closed after processing.
        """

        self.binary_mask[self.side_label_key.get()] = bbox("per",np.array(self.image[self.side_label_key.get()]), [self.start_x,self.start_y,self.end_x,self.end_y], self.page_ctrl.shared_data['segment_model']) # use model to segment mask based on area that user select
        self.binary_mask[self.side_label_key.get()] = (self.binary_mask[self.side_label_key.get()].squeeze()).astype(np.uint8) # If the array has more than one singleton dimension, .squeeze() will remove all of them
        self.binary_mask[self.side_label_key.get()] = cv2.cvtColor(self.binary_mask[self.side_label_key.get()], cv2.COLOR_GRAY2BGR) # change color to match opencv color format
        self.binary_mask[self.side_label_key.get()] = self.resize_mask(self.binary_mask[self.side_label_key.get()]) # add border to segment mask result
        # overlay_image = ((self.binary_mask[self.side_label_key.get()]) * np.array(self.image[self.side_label_key.get()])) + np.array(self.image[self.side_label_key.get()])
        overlay_image = np.where(self.binary_mask[self.side_label_key.get()][:, :, :] == 255, [0,0,0],  np.array(self.image[self.side_label_key.get()])) # overlay mask onto image
        self.seg_pil_image = Image.fromarray(overlay_image.astype('uint8'))
        self.seg_tk_img = ImageTk.PhotoImage(self.seg_pil_image)
        self.seg_canvas.config(width=self.image[self.side_label_key.get()].width, height=self.image[self.side_label_key.get()].height)
        self.seg_canvas_image = self.seg_canvas.create_image(0, 0, anchor="nw", image=self.seg_tk_img)
        self.after(0, self.loading.close) # close process popup after finish

    def resize_mask(self,mask:np.ndarray, scale:int=11) -> np.ndarray:
        """
        Resizes the given mask by dilating it with an elliptical kernel.
        Args:
            mask (numpy.ndarray): The input mask to be resized (typically a binary image).
            scale (int, optional): The size of the elliptical kernel used for dilation. Defaults to 11.
        Returns:
            numpy.ndarray: The dilated (resized) mask.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale)) 
        dilated_mask = cv2.dilate(mask, kernel)
        return dilated_mask

    def next_step(self):
        """
        Advances the application to the next step by saving the current binary mask to shared data,
        marking the mask as edited, and displaying the 'ImageProcessingApp' frame.
        """
        self.page_ctrl.shared_data["segment_mask"] = self.binary_mask # save binary image to shared data
        self.page_ctrl.shared_data["mask_edited"] = True # save edited status to shared data
        self.page_ctrl.show_frame('ImageProcessingApp') # change to main page (process page)

    def next_image(self):
        """
        Advances to the next image in the navigation sequence.
        Updates the navigation index, enables or disables navigation buttons based on the current position,
        and updates the canvas to display the new image.
        """
        key, _ = self.nav_image.next() # get next side view
        current = self.nav_image.index # get current side view
        if(current >= len(list(self.image.keys()))): # if current image is last on dict
            self.btn_right.config(state=tk.DISABLED) # disabled next image
            self.btn_left.config(state=tk.NORMAL)
        else: # otherwise normal
            self.btn_right.config(state=tk.NORMAL)
            self.btn_left.config(state=tk.NORMAL)
        self.set_canvas(key)


    def previous_image(self):
        """
        Navigates to the previous image in the image navigator.
        Updates the navigation buttons' states based on the current image index:
            - Disables the left button if at the first image.
            - Enables both left and right buttons otherwise.
        Sets the canvas to display the newly selected image.
        """
        key, _ = self.nav_image.previous() # get previous side image
        current = self.nav_image.index # get current index
        if(current == 0): # if at first image 
            self.btn_right.config(state=tk.NORMAL)
            self.btn_left.config(state=tk.DISABLED) # disabled left button
        else: # otherwise normal
            self.btn_right.config(state=tk.NORMAL)
            self.btn_left.config(state=tk.NORMAL)
        self.set_canvas(key)

    def back_step(self):
        """
        Navigates back to the 'ImageProcessingApp' frame by instructing the page controller to display it 
        without save changes.
        """

        self.page_ctrl.show_frame('ImageProcessingApp')


    def build_left(self):
        """
        Builds and configures the left section of the UI.
        This method creates and packs a label titled "Input" with specific styling
        into the left frame, followed by packing the input canvas with padding.
        """
        label = tk.Label(self.left_frame, text="Input", font=("Arial", 14), justify="center", fg="green", bg="white") # head of frame 
        label.pack(anchor="c", padx=10, pady=10) # adjust to center
        self.input_canvas.pack(padx=10, pady=10) # show canvas image

    def build_center(self):
        """
        Creates and packs a series of buttons into the center frame of the UI.
        Each button is associated with a specific action, such as loading an image, loading a configuration,
        drawing a rectangle mask, running segmentation, saving the image or configuration, clearing the mask,
        or proceeding to the next step. The actions are bound to the buttons using lambda functions to ensure
        they are executed only when the corresponding button is clicked.
        """
        buttons = ["Load new image", "Load config", "Rectangle Mask", "Segment", "Save image", "Save config", "Clear", "Next step"] # button name
        # button action
        bt_action = [lambda: self.load_mask(), lambda:self.reload_config(), lambda: self.direct_draw() ,lambda: self.run_segment(),lambda: self.save_mask(), lambda: self.save_config() ,lambda: self.clear_mask(), lambda: self.next_step()] # using lambda: ..., delay execution — the code inside runs only when the button is clicked. 
        for text,cmd in zip(buttons,bt_action): # displayed by order as column
            btn = tk.Button(self.center_frame, text=text,command=cmd, width=15)
            btn.pack(pady=10)

    def build_right(self):
        """
        Constructs and arranges the widgets on the right frame of the UI.
        This method creates a label titled "Mask" with specific styling and packs it into the right frame.
        It also packs the segmentation canvas (`self.seg_canvas`) below the label with padding.
        Assumes that `self.right_frame` and `self.seg_canvas` are already initialized.
        """
        label = tk.Label(self.right_frame, text="Mask", font=("Arial", 14) , justify="center", fg="green", bg="white") # head of frame
        label.pack(anchor="c", padx=10, pady=10) # adjust to center
        self.seg_canvas.pack(padx=10, pady=10) # displayed overlay image

    def build_header(self):
        """
        Builds the header section of the UI with navigation buttons and a label.
        This method configures the header_frame to have three columns of equal weight,
        and adds the following widgets:
            - A left navigation button ('<') in the first column to go to the previous image.
            - A label in the center column displaying the current side or status.
            - A right navigation button ('>') in the third column to go to the next image.
        The buttons and label are styled with a green background and white foreground.
        """
        # Configure 3 columns in grid ------------------------
        self.header_frame.grid_columnconfigure(0, weight=1) # left
        self.header_frame.grid_columnconfigure(1, weight=1) # center 
        self.header_frame.grid_columnconfigure(2, weight=1) # right
        # --------- left grid ------------------
        self.btn_left = tk.Button(self.header_frame, text='<',command=lambda: self.previous_image(), width=15, bg="green", fg="white") # action: go to previous side image
        self.btn_left.grid(row=0, column=0, sticky="w") # stick to left (west)
        # --------- center grid ------------------
        side_label = tk.Label(self.header_frame, textvariable=self.side_label_show, font=("Arial", 14), bg="green", fg="white") # show current side name
        side_label.grid(row=0, column=1)
        # --------- right grid ------------------
        self.btn_right = tk.Button(self.header_frame, text='>',command=lambda: self.next_image(), width=15, bg="green", fg="white") # action: go to next side image
        self.btn_right.grid(row=0, column=2, sticky="e") # stick to right (east)


