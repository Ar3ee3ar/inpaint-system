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
import time
import datetime
import numpy as np
# from ultralytics import YOLO
from typing import Union

# from .mask_page import MaskApp
from page.loadingPopup_page import LoadingPopup
from page import get_file_path, find_file_path

sys.path.append("..")
from tools.utils import find_fol, create_fol, check_corrupt
from tools.overlay import OverlayCube, OverlayGAN
from tools.pix2pix import Generator as p2pG
from tools.pix2pix import Discriminator as p2pD
from tools.deepfill.inpaint_model import InpaintGenerator
from tools.sam_tools.segment_anything import sam_model_registry


event_obj = threading.Event()

# ----------------- initial model setting ---------------
def load_model_generator(model_type:str, model_input_2d:Union[list, tuple], model_weight:str) -> object:
    """
    Loads and initializes an inpainting model generator based on the specified model type.
    Args:
        model_type (str): The type of model to load. Supported values are "p2p" for Pix2Pix and "df" for DeepFill.
        model_input_2d (Union[list, tuple]): The input shape for the model as a list or tuple of two integers [height, width].
        model_weight (str): The path or identifier for the model weights file.
    Returns:
        generator: The initialized and weight-loaded model generator instance.
    Raises:
        FileNotFoundError: If the specified model weights file cannot be found.
        ValueError: If an unsupported model_type is provided.
    Note:
        - For "p2p", the generator is initialized with the specified input shape and weights.
        - For "df", a dummy forward pass is performed before loading weights to initialize the model.
    """
    keras.backend.clear_session()
    # yolo_model = YOLO("yolov8n.pt")
    if(model_type == "p2p"):
        generator = p2pG.Generator(input_shape=[model_input_2d[0],model_input_2d[1],3])
        generator.load_weights(find_file_path(model_weight)) # 500
    elif(model_type == "df"):
        generator = InpaintGenerator()
        dummy_input = tf.zeros([1, 512, 512, 3])
        dummy_mask = tf.zeros([1, 512, 512, 1])
        _, _, _ = generator(dummy_input,dummy_mask, training=False)
        generator.load_weights(find_file_path(model_weight))
    else:
        raise ValueError(f": unsupported model_type is provided : {model_type}")
    print('inpainting model start success')
    return generator

def load_model_segment() -> object:
    """
    Loads and initializes the segmentation model using the specified checkpoint file.
    Returns:
        object: The loaded segmentation model instance.
    Raises:
        FileNotFoundError: If the checkpoint file is not found.
        Exception: If the model fails to load for any other reason.
    Side Effects:
        Prints status messages to the console indicating the progress of model loading.
    """
    print('start segmentation model')
    sam = sam_model_registry["vit_l"](checkpoint=find_file_path("weight_sam/sam_vit_l_0b3195.pth")).to(device=torch.device('cpu'))
    print('segmentation model start success')
    return sam

# def load_yolo_model():
#     print('start object detection model')
#     model = YOLO("yolov8n.pt")
#     print('object detection model start success')
#     return model

# ------------------ precessing page --------------------------
class ImageProcessingApp(tk.Frame):
    """
    A Tkinter-based GUI application for image processing tasks, including inpainting and segmentation,
    with support for multiple configurations and progress tracking.
    Features:
        - Folder selection for input images.
        - Configuration selection via dropdown, with dynamic loading from JSON.
        - Option to overwrite existing processed images.
        - Progress bar and status updates during processing.
        - Mask selection and editing for inpainting tasks.
        - Threaded processing to keep the UI responsive.
        - Error handling and user feedback via message boxes.
        - Integration with external models for segmentation and inpainting.
    Args:
        parent (tk.Widget): The parent widget.
        controller: The controller object managing page transitions and shared data.
        configs (dict): Dictionary of available configuration settings.
    Attributes:
        page_ctrl: Reference to the controller for page navigation and shared data.
        config_data (dict): The currently selected configuration data.
        configs (dict): All available configurations.
        type_process_scale (str): Processing mode, either 'resize' or 'pad_image'.
        have_mask_cube (dict): Status of mask availability for each cube face.
        have_mask_top (bool): Status of top mask availability.
        dir_entry (tk.Entry): Entry widget for folder path.
        message_config (tk.StringVar): Status message for configuration loading.
        selected_config (tk.StringVar): Currently selected configuration name.
        dropdown (ttk.Combobox): Dropdown for selecting configuration.
        switch_button (tk.Button): Button to start mask segmentation.
        overwrite_var (tk.StringVar): Radiobutton variable for overwrite option.
        run_button (tk.Button): Button to start processing.
        cancle_button (tk.Button): Button to cancel processing.
        progress_per (tk.IntVar): Progress percentage for the progress bar.
        progress (tk.StringVar): Progress text (e.g., "3/10").
        progress_folder (tk.StringVar): Current folder being processed.
        photo (ImageTk.PhotoImage): Loaded image for display in the UI.
    Methods:
        start_segment(): Handles mask segmentation initiation.
        start_thread_state_setting(config_name, status_setting): Starts a thread to set up models/configs.
        state_setting(config_name, status_setting): Loads models and updates shared data.
        set_initial_config(config_name): Sets and initializes the selected configuration.
        refresh_config(event=None): Updates configuration when dropdown changes.
        refresh_file_config(event): Reloads configuration file and updates dropdown.
        browse_dir(): Opens a dialog to select the input folder.
        hide_button(widget): Hides a widget from the UI.
        show_button(widget): Shows a previously hidden widget.
        stop_process(): Stops the current processing thread.
        run_process(): Starts processing images (single-threaded).
        start_thread_image(): Starts image processing in a new thread.
        start_processing(directory, overwrite): Main processing logic for images, with progress updates.
    """

    def __init__(self,parent,controller,configs):
        super().__init__(parent)
        self.setting_path = 'system_settings.json'
        self.page_ctrl = controller
        self.config_data = {}  # <-- This will store the current config
        self.configs = configs # All available config setting
        self.type_process_scale = 'resize' # 'pad_image' or 'resize'
        self.have_mask_cube = { # Bool value if each map have mask
            "cube_front": False,
            "cube_right": False,
            "cube_back": False,
            "cube_left": False,
            "cube_top": False,
            "cube_bottom": False
        }
        self.have_mask_top = False # if bird view have mask
        
        # Crate frame for contain all UI
        main_frame = tk.Frame(self, bd=0.5, relief="solid", highlightbackground="green", highlightcolor="green", highlightthickness=0.5) # frame with green border
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Setting main font to apply for all object
        self.option_add("*Font", ("Kanit", 10))

        # Frame : select folder --------------------------------------------------
        dir_frame = tk.Frame(main_frame)
        dir_frame.pack(fill="x", pady=(20, 5))

        tk.Label(dir_frame, text="เลือก Folder :").pack(side="left", padx=5)
        self.dir_entry = tk.Entry(dir_frame, width=60) # Entry : show selected folder
        self.dir_entry.pack(side="left", padx=5)
        tk.Button(dir_frame, text="เลือก", command=self.browse_dir, bg="lightblue", fg="black").pack(side="left", padx=5) # button for call pop up page for selected folder

        # Frame : select config setting -----------------------------------------------------
        config_status_frame = tk.Frame(main_frame) # frame for contain config setting
        config_status_frame.pack(fill="x", pady=5)

        tk.Label(config_status_frame, text="เลือก Config : ").pack(side="left", padx=5)

        self.message_config = tk.StringVar() # variable for contain status for config reading to notify user
        self.message_config.set('')
        # ------------ Drop down show all config setting -----
        self.selected_config = tk.StringVar()
        self.dropdown = ttk.Combobox(config_status_frame, textvariable=self.selected_config, values=list(self.configs.keys()), state="readonly")
        self.dropdown.pack(side="left", padx=5)
        self.selected_config.set(list(self.configs.keys())[0]) # set default config to first config
        # -------------- Bind the selection event ----------------
        self.dropdown.bind("<<ComboboxSelected>>", self.refresh_config) # refresh config list on click drop down to show all config
        self.dropdown.bind("<Button-1>", self.refresh_file_config) # left click to set wanted config
        # ----- Button: switch page to mask_page (in same frame with config setting area) --------------------------
        # ---- if directory is not selected button will disabled
        self.switch_button = tk.Button(config_status_frame, text="mask รถ", command=lambda: self.start_segment(), bg="white", fg="black", state=tk.DISABLED) 
        self.switch_button.pack(side="left", padx=15)
        tk.Label(config_status_frame,textvariable = self.message_config).pack(side="left", padx=5) # place here for show after mask cars button
        # Frame : overwrite setting radio button --------------------------
        dir_frame2 = tk.Frame(main_frame) # create frame for overwrite setting
        dir_frame2.pack(fill="x", pady=5)
        tk.Label(dir_frame2, text="ประมวลผลภาพใหม่ทั้งหมด :").pack(side="left", padx=5, anchor="w")
        
        # ----- Radiobutton variable ----------------------------------------
        self.overwrite_var = tk.StringVar(value="False")
        tk.Radiobutton(dir_frame2, text="ใช่ (True)", variable=self.overwrite_var, value="True").pack(side="left", padx=5)
        tk.Radiobutton(dir_frame2, text="ไม่ใช่ (False)", variable=self.overwrite_var, value="False").pack(side="left", padx=5)

        # Button : Start processing (in same frame with overwrite setting) ---------------------
        self.run_button = tk.Button(dir_frame2, text="เริ่มประมวลผล", command=lambda: self.start_thread_image(), bg="green", fg="white") # start thread for inpainting process
        self.run_button.pack(side="left", padx=15)

        # Button : Cancel process (in same frame with overwrite setting) ---------------------
        self.cancle_button = tk.Button(dir_frame2, text="ยกเลิก", command=lambda: self.stop_process(), bg="red", fg="white")
        self.cancle_button.pack(side="left", padx=15)
        self.hide_button(self.cancle_button) # hide button when not start process

        # Frame : Progress notify area ------------------------------------
        dir_frame3 = tk.Frame(main_frame)
        dir_frame3.pack(pady=20, padx=5, anchor="c")

        # ----------- Progress bar widget -------------------------
        self.progress_per = tk.IntVar() # int for setting in progress bar
        self.progress = tk.StringVar() # process file / all file
        self.progress_folder = tk.StringVar() # current folder that program process
        self.progress_folder.set('') # initial empty value for folder
        self.progress.set('0/0') # initial zero for count progress file 
        tk.Label(dir_frame3,textvariable = self.progress_folder).pack(side="left") # place for show processing folder
        self.progressbar = ttk.Progressbar(dir_frame3,variable=self.progress_per,length=400).pack(side="left") # place for progress bar
        tk.Label(dir_frame3,textvariable = self.progress).pack(side="left") # place for count progress file (str)

        # show flow inpainting process (in same frame as main frame) ---------------------
        image_path = "image_app/flow.png"
        
        try:
            image = Image.open(find_file_path(image_path))
            image = image.resize((450, 250))
            self.photo = ImageTk.PhotoImage(image)
            self.image_label = tk.Label(main_frame, image=self.photo)
            self.image_label.pack(pady=10)
        except Exception as e:
            print(f"Error loading image: {e}")

        # show text ------------------------------------------------------
        tk.Label(main_frame, text="โครงการพัฒนาแบบจำลองภูมิประเทศความละเอียดสูงเพื่อการบริหารจัดการน้ำ และแก้ไขปัญหาอุทกภัยอย่างยั่งยืน",
                 wraplength=480, justify="center", fg="green").pack(side="bottom", pady=10)
        # set initial setting for inpainting model (eg. Call model, set config mask, set config variable)
        self.set_initial_config(self.selected_config.get())
    
    def change_setting(self):
        with open(self.setting_path, "r") as f:
            self.configs = json.load(f)

    def start_segment(self):
        """
        Initiates the image segmentation process by validating the input directory and image files.
        This method performs the following steps:
        1. Retrieves the directory path from the user input.
        2. Searches for image files (.jpg) in the specified directory.
        3. Checks if the process is not already running.
        4. Validates the existence and integrity of the first image file found.
        5. Updates shared data and UI elements if a valid image is found.
        6. Handles and displays error messages for various failure cases:
            - No images found in the directory.
            - Corrupted image file detected.
            - Input directory not found.
            - Process is already running.
        Raises:
            Displays error dialogs for invalid input, missing files, or process conflicts.
        """
        directory = self.dir_entry.get() # get directory path from dir_entry 
        path_input = find_fol(directory) # get all folder that have 'Camera' folder in it
        # if not in processing flow do :
        if(self.run_button["text"] != "กำลังประมวลผล"):
            # try except: if not found Camera folder raise error
            try:
                dir_list_input = glob.glob(path_input[0]+'*'+'.jpg') # get image in Camera folder of input directory 
                # if not found img in Camera: raise message box 
                if(len(dir_list_input) > 0):
                    # if image not corrupt set shared_data (image path, select config name, dict of select config, process type) and go to mask_page
                    if(not check_corrupt(dir_list_input[0])):
                        self.page_ctrl.shared_data["input_img"] = dir_list_input[0] # image path 
                        self.page_ctrl.shared_data["config_name"] = self.selected_config.get() # select config name
                        self.page_ctrl.shared_data["config_data"] = self.config_data # dict of select config
                        self.page_ctrl.shared_data["type_process_scale"] = self.type_process_scale # process type
                        self.run_button.config(text="เริ่มประมวลผล", bg="green",state="normal") # change to start phase after view mask
                        self.update_idletasks()  # Update UI (immediately change)
                        self.page_ctrl.show_frame('MaskApp') # switch page
                    # if found corrupt image show message box
                    else:
                        print('พบภาพที่ไฟล์เสียหายในโฟลเดอร์')
                        messagebox.showerror("Error", "พบภาพที่ไฟล์เสียหายในโฟลเดอร์")
                else:
                    print('ไม่มีภาพในโฟลเดอร์')
                    messagebox.showerror("Error", "ไม่มีภาพในโฟลเดอร์")
            except IndexError: # raise error for notify error of state of selected folder (not found image folder)
                print('ไม่พบโฟลเดอร์เก็บภาพ (Camera)')
                messagebox.showerror("Error", "ไม่พบโฟลเดอร์เก็บภาพ (Camera)")
        else:
            messagebox.showerror("Error", "โปรดยกเลิกการประมวลผลก่อน")
    
    def confirm_and_exit(self, error_type, error_msg):
        full_msg = f"{error_type} occurred:\n{error_msg}\n\nDo you want to close the program?"
        answer = messagebox.askyesno(error_type, full_msg)
        if answer:
            self.page_ctrl.destroy()
            sys.exit(1)
        # answer = messagebox.askyesno("Memory Error", "Memory allocation failed.\nDo you want to close the program?")
        # if answer:
            # self.page_ctrl.destroy()
            # sys.exit(1)

    def start_thread_state_setting(self,config_name:str, status_setting:str):
        """
        Starts a new thread to execute the state_setting method with the given configuration name and status setting.
        Displays a loading popup while the operation is in progress.
        Args:
            config_name (str): The name of the configuration to be used in the state_setting method.
            status_setting (str): The status setting to be applied (eg. 'initial', 'update')
        """
        self.loading = LoadingPopup(self, "Processing...") # show popup box for processing bar
        threading.Thread(target=self.state_setting, daemon=True, args=(config_name,status_setting)).start() # start thread

    def state_setting(self,config_name:str, status_setting:str):
        """
        Configures and initializes the state for the main page based on the provided configuration name and status.
        This method performs the following actions:
        - Updates the UI to indicate the start of the configuration process.
        - Initializes overlay classes (`OverlayGAN` and `OverlayCube`) with the specified configuration.
        - Loads and sets the model generator for both overlay classes.
        - Enables mask usage for both overlay classes.
        - If `status_setting` is 'initial', loads the segmentation model and stores it in shared data.
        - Resets the 'mask_edited' flag in shared data.
        - Updates the UI to indicate completion of the configuration process.
        - Closes the loading popup safely from the main thread.
        Args:
            config_name (str): The name of the configuration to load and apply.
            status_setting (str): The status indicating the type of setting operation (e.g., 'initial').
        Returns:
            None
        """
        try:
            self.message_config.set('Start read '+config_name+' config...') # set message config to show status
            self.update_idletasks()  # Update UI (immediately change)
            # ------- setting config to class (bird view inpainting) ---------------
            print("+++ top view setting +++")
            self.over_class = OverlayGAN(self.config_data,config_name,self.type_process_scale) # set __init__ to bird view overlay
            generator = load_model_generator(self.config_data['model_type'],self.config_data["model_input"],self.config_data["model_weight"]) # load generator model
            self.over_class.set_model_generator(generator) # set model to class
            self.over_class.set_useMask() # set mask in config setting
            # ------- setting config to class (cube map/ side view inpainting) ---------------
            print("+++ side view setting +++")
            self.over_cube_class = OverlayCube(self.config_data,config_name,self.type_process_scale) # set __init__ to side view overlay
            self.over_cube_class.set_model_generator(generator) # load generator model
            self.over_cube_class.set_useMask() # set mask in config setting
            # --------- only load segmentation model in starting phase (1st open program) --------------
            if(status_setting == 'initial'):
                segment_model = load_model_segment()
                self.page_ctrl.shared_data['segment_model'] = segment_model # set segment model for using in mask_page.py
            self.page_ctrl.shared_data['mask_edited'] = False # reset mask edited status (select other config need to reset former state)
            self.message_config.set('Complete setting '+config_name)
            self.update_idletasks()  # Update UI (immediately change)
            print("+++ update setting complete +++")
            self.after(0, self.loading.close)  # Close 'processing...' popup safely from main thread after complete thread
        except np.core._exceptions._ArrayMemoryError as e:
            print("Memory error occurred:", e)
            # Must call GUI code from main thread
            self.after(0, lambda: self.confirm_and_exit("Memory Error", "Memory allocation failed"))
        except Exception as e:
            error_msg = e
            print("Error:", error_msg)
            self.after(0, lambda: self.confirm_and_exit("Error", error_msg))

    def set_initial_config(self, config_name:str):
        """
        Sets the initial configuration for the application based on the provided configuration name.
        If the specified configuration name exists in the available configurations, it selects and applies
        that configuration. Otherwise, it defaults to the first available configuration. The method also
        updates the internal state and starts a thread to handle additional state settings.
        Args:
            config_name (str): The name of the configuration to set as initial.
        Side Effects:
            - Updates `self.selected_config` to the chosen configuration name.
            - Updates `self.config_data` with the corresponding configuration data.
            - Initiates a thread to handle state setting for the selected configuration.
        """
        if config_name in self.configs:
            self.selected_config.set(config_name) # set selected config name
            self.config_data = self.configs.get(config_name, {})  #  set selected config dic
            self.start_thread_state_setting(config_name,'initial') # start thread to setting variable
        else:
            self.selected_config.set(list(self.configs.keys())[0]) # set selected config name (1st element as default)
            self.config_data = self.configs.get(list(self.configs.keys())[0], {})  #  set selected config dic (1st element as default)
            self.start_thread_state_setting((self.configs.keys())[0],'initial') # start thread to setting variable
    
    # Refresh content when dropdown changes
    def refresh_config(self,event=None):
        """
        Refreshes the current configuration based on the selected configuration name.
        Retrieves the selected configuration name from the UI, updates the internal configuration data,
        and initiates a thread to update the state settings accordingly.
        Args:
            event (optional): The event object, if triggered by a UI event. Defaults to None.
        """
        if(self.run_button["text"] != "กำลังประมวลผล"):
            config_name = self.selected_config.get() # set selected config name
            self.config_data = self.configs.get(config_name, {}) #  set selected config dic
            self.start_thread_state_setting(config_name,'update') # start thread to setting variable (update)
    
    def refresh_file_config(self,event):
        """
        Refreshes the application's configuration by reloading settings from the 'config_gan_ui.json' file.
        This method updates the internal configuration dictionary, refreshes the dropdown menu with new configuration keys,
        and resets the run button to its initial state. It also ensures that UI changes are immediately reflected.
        Args:
            event: The event object that triggered this method, typically from a UI action.
        """
        if(self.run_button["text"] != "กำลังประมวลผล"):
            # reread config json file ---------------------------
            config_path = find_file_path("config_gan_ui.json")
            with open(config_path, "r") as f:
                self.configs = json.load(f)
            # ------------------------------------------------
            self.dropdown["values"] = list(self.configs.keys()) # set new list for drop down list
            self.run_button.config(text="เริ่มประมวลผล", bg="green",state="normal") # change to start phase after select new config setting
            self.update_idletasks()  # Update UI (immediately change)

    def browse_dir(self):
        """
        Opens a directory selection dialog for the user to choose a folder. 
        If the process is not currently running (as indicated by the run_button's text), 
        updates the directory entry field with the selected folder path, resets the run button's 
        text and color, enables the switch button, and updates the UI immediately. 
        If the process is running, shows an error message prompting the user to cancel the process first.
        """
        if(self.run_button["text"] != "กำลังประมวลผล"):
            folder_selected = filedialog.askdirectory()
            if folder_selected:
                self.dir_entry.delete(0, tk.END)
                self.dir_entry.insert(0, folder_selected)
                self.run_button.config(text="เริ่มประมวลผล", bg="green",state="normal")
                self.switch_button.config(state=tk.NORMAL)
                self.update_idletasks()  # change UI (immediately change)
        else:
            messagebox.showerror("Error", "โปรดยกเลิกการประมวลผลก่อน")

    def hide_button(self,widget):
        """
        Hides the specified widget by removing it from its parent container using `pack_forget()`.
        Args:
            widget: The Tkinter widget to be hidden.
        """
        # This will remove the widget from toplevel
        widget.pack_forget()

    # Method to make Button(widget) visible
    def show_button(self,widget):
        """
        Displays the given widget as a button by packing it to the left side of its parent widget with horizontal padding.
        Args:
            widget (tk.Widget): The widget to be displayed as a button.
        """
        widget.pack(side="left", padx=15)

    def stop_process(self):
        """
        Stops the currently running process by setting the event object and hiding the cancel button.
        This method signals the associated process or thread to stop by setting a predefined event object (`event_obj`).
        It also hides the cancel button in the user interface to prevent further cancellation attempts.
        """
        print('************************ stop process thread ************************')
        event_obj.set()
        self.hide_button(self.cancle_button)

    def run_process(self):
        """
        Handles the execution of the processing workflow triggered by the user.
        Retrieves the selected directory and overwrite option from the UI, updates the run button to indicate processing,
        and validates the directory input. If a directory is selected, initiates the processing by calling
        `start_processing` with the provided parameters. Displays an error message if no directory is selected.
        """
        directory = self.dir_entry.get()
        overwrite = self.overwrite_var.get()
        self.run_button.config(text="กำลังประมวลผล", bg="orange", state="disabled", relief="flat", disabledforeground="black")
        self.update_idletasks()  # change UI (immediately change)
        if not directory:
            messagebox.showerror("Error", "Please select a directory.")
            return
        
        self.start_processing(directory, overwrite)

    def start_thread_image(self):
        """
        Starts a new thread to perform the inpainting process, handling UI updates and input validation.
        This method checks for required masks and directory selection, updates the UI to reflect the processing state,
        and disables relevant controls. If all validations pass, it spawns a new daemon thread to execute the
        long-running inpainting task.
        Steps performed:
            - Validates that a directory is selected.
            - Checks if at least one mask (top or cube faces) is set.
            - Updates UI elements to indicate processing (button text, progress bars, disables controls).
            - Starts a new thread to run the inpainting process.
        Raises:
            Shows a message box with an error if required inputs are missing.
        """
        # if not found directory show error message box
        directory = self.dir_entry.get() # get directory 
        overwrite = self.overwrite_var.get() # get overwrite
        if not directory:
            messagebox.showerror("Error", "Please select a directory.")
            return
        # setting process state for UI --------------------------------
        self.run_button.config(text="กำลังประมวลผล", bg="orange", state="disabled", relief="flat", disabledforeground="black") # change text on button to 'processing'
        self.show_button(self.cancle_button) # show cancel button for stop process
        self.progress.set('0/0') # set initial count progress file
        self.progress_folder.set('') # set initial process folder
        self.progress_per.set(0) # set progress bar to start
        self.update_idletasks()  # change ui (immediately change)

        # ---------- start inpainting thread ----------------------------
        print('************************ start thread ************************')
        # Disable 'mask button' and 'config dropdown' --------------
        self.switch_button.config(state=tk.DISABLED)
        self.dropdown.config(state=tk.DISABLED)
        # start thread --------------------------------------------
        thread = threading.Thread(target=self.start_processing, daemon=True, args=(directory,overwrite))
        # thread.daemon = True  # Daemon thread will exit when the main program exits
        thread.start() # start overlay thread
        # --------------------------------------------------------------
    def check_mask(self):
        keys_to_check = ["cube_front","cube_right","cube_back","cube_left","cube_top","cube_bottom"]
        # Get keys as a list
        keys = list(self.page_ctrl.shared_data["segment_mask"].keys())
        # Define range that overlay function want
        start = keys.index("cube_front")
        end = keys.index("cube_bottom")
        # if edit mask on mask_page set new mask
        if(self.page_ctrl.shared_data["mask_edited"]):
            self.over_class.set_segMask(self.page_ctrl.shared_data["segment_mask"]["top_view"])
            selected = {k: self.page_ctrl.shared_data["segment_mask"][k] for k in keys[start:end + 1]}
            self.over_cube_class.set_segMask(selected)

        mask_top_setting = self.over_class.get_haveMask_status() # get status of mark in bird view (currently have mask or not)
        mask_cube_setting = self.over_cube_class.get_haveMask_status() # get status of mark in cube map (currently have mask or not)
        # if any mask is not initiate to class show error message box
        if(not mask_top_setting) and (not any(mask_cube_setting[key] for key in keys_to_check)):
            messagebox.showerror("Error", "Please select mask for inpainting")
            return
        
    def start_processing(self, directory:str, overwrite:bool):
        """
        Starts the processing of Camera folder in the specified directory.
        This method iterates through all found Camera (folders) in the given directory,
        processes each image file within those Camera folder, and saves the output to a new folder.
        It supports skipping already processed files unless the overwrite flag is set as True.
        The method also updates the UI to reflect progress and handles user interruption.
        Args:
            directory (str): The path to the directory containing the albums to process.
            overwrite (bool): If True, existing output files will be overwritten. If False, already processed files will be skipped.
        Side Effects:
            - Updates UI elements to show progress and status.
            - Creates output folders as needed.
            - Generates error reports for corrupted files.
            - Handles user interruption via an event object.
        Notes:
            - Assumes the existence of several helper functions and class attributes, such as `find_fol`, `create_fol`, `check_corrupt`, `self.over_class`, and `self.over_cube_class`.
            - Progress and status are shown via UI elements and printed to the console.
            - Error reports are saved in the output folder if corrupted files are found.
        """
        self.check_mask()
        # Clear the event flag before starting processing to ensure a fresh state
        try:
            event_obj.clear()
            # get important variable ------------------------------
            print(f"Directory: {directory}")
            print(f"Overwrite: {overwrite}")     
            path_input = find_fol(directory) # find folder that have Camera folder in its
            path_input.sort() # sort folder 
            print("all path_input: ")
            for print_path in path_input: # show all folder that have Camera (get from path_input)
                print(os.path.basename(os.path.dirname(os.path.dirname(print_path))))
            # run through each folder ---------------------------------------
            for i in range(len(path_input)): # run through each album
                # if thread is stop process after user cancel process -----------
                if event_obj.is_set():
                    self.run_button.config(text="หยุดการทำงาน", bg="red", state="disabled", relief="flat", disabledforeground="black") # change text of run button 
                    self.switch_button.config(state=tk.NORMAL) # enable mask cars (switch to mask_page) 
                    self.dropdown.config(state="readonly") # enable config setting
                    self.update_idletasks()  # change UI (immediately change)
                    break
                print('--------------------------------------------')
                # inpainting process --------------------------------------
                main_path = path_input[i].replace('\\','/') # replace \\ in path (found in windows os)
                print(f"path_input: {main_path}")
                line_name = os.path.basename(os.path.dirname(os.path.dirname(main_path))) # get road line name (text before last folder)
                print(f"line_name: {line_name}")
                self.progress_folder.set(line_name) # set current process folder
                self.update_idletasks()  # change UI (immediately change)
                
                save_path = main_path + line_name + '_output' # set output folder path

                target = [] # list for target name (as number(int) of file name - img_'000000'.jpg)
                normal_file = 0 # variable for normal file after process
                corrupted_file = 0 # variable for corrupted file
                dir_list_input = glob.glob(main_path+'*'+'.jpg') # get image file
                dir_list_input.sort() # sort file
                if(not os.path.exists(save_path)): # if not found output folder -> create folder
                    create_fol(save_path)
                else:
                    if(not bool(strtobool(overwrite))): # if not overwrite file
                        for item in dir_list_input[:]: # run through image file path
                            pic_full_name = (item.replace('\\','/')).split('/')[-1] # get image name
                            pic_name = save_path+'/'+pic_full_name[:-4]+'_O.jpg' # set output format to find if putput already exist 
                            if(os.path.exists(pic_name)): # check if output image file in directory
                                dir_list_input.remove(item) # remove from list to process file
                # if still have input image path to process do:
                if(len(dir_list_input)>0):
                    time_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # get current date time
                    report_file = save_path+'/'+line_name+'_error_report_'+time_now+'.txt' # define report file name (roadLine_error_report_%Y%m%d-%H%M%S.txt)
                    # ---------- get number from image name -----------------
                    for j in range(len(dir_list_input)):
                        name_file = ((dir_list_input[j].replace('\\','/')).split('/'))[-1] # get file name
                        num_file = ((name_file.split('_'))[1])[:-4] # get 000000.jpg -> 000000
                        target.append(int(num_file)) # stored int(number from file name)
                    #------- run through target (img file that need to process) --------------
                    for count_pic, coor in enumerate(target):
                        # if thread is stop process after user cancel process -----------
                        if event_obj.is_set():
                            self.run_button.config(text="หยุดการทำงาน", bg="red",state="disabled", relief="flat", disabledforeground="black")
                            self.switch_button.config(state=tk.NORMAL)
                            self.dropdown.config(state="readonly")
                            self.update_idletasks()  # change UI (immediately change)
                            break
                        # ---- inpaint image ver. gps ---------------------------------
                        # source_pic = self.over_class.find_relate_new(coor_detail,coor,"utm")
                        # self.over_class.overlay(main_path,save_path,source_pic,coor,str(coor),coor_detail)
                        # ---- inpaint image ver. model ---------------------------------
                        keys_to_check = ["cube_front","cube_right","cube_back","cube_left","cube_top","cube_bottom"] # define cube map side
                        # if not corrupted image file do below  | if corrupted show error message and write to txt file
                        if(not check_corrupt(main_path+ "img_"+str(coor).rjust(6, '0')+".jpg",report_file)):
                            # check for bird view mask status in class -------------------
                            if((self.over_class.get_haveMask_status())): # if found mask do bird view inpainting first
                                final_pano_output = self.over_class.overlay(main_path,coor) # inpainting bird view then stored in 'final_pano_output'
                                if (any((self.over_cube_class.get_haveMask_status())[key] for key in keys_to_check)): # if found mask on any side of cube map
                                    final_cube_output = self.over_cube_class.overlay(final_pano_output,coor) # inpainting cube map then stored in 'final_pano_output'
                                    self.over_cube_class.save_pano(final_cube_output,save_path,coor) # save final panorama result (bird + cube)
                                else:
                                    self.over_class.save_pano(final_pano_output,save_path,coor) # save final panorama result (bird)
                            # not found mask on bird view. So, check for side view
                            else:
                                if (any((self.over_cube_class.get_haveMask_status())[key] for key in keys_to_check)):
                                    final_cube_output = self.over_cube_class.overlay(main_path,coor)
                                    self.over_cube_class.save_pano(final_cube_output,save_path,coor)
                            normal_file = normal_file + 1 # count process file
                        else: # file is corrupt
                            corrupted_file = corrupted_file + 1 # count corrupt file
                        # ----------- update progress result ------------------
                        percent_num = (int(count_pic+1)*100)/len(target) # calculate precent of processed file
                        self.progress.set(str(count_pic+1)+'/'+str(len(target))) # set processed file / all file need to procrss
                        self.progress_per.set(int(percent_num)) # set value to progress bar
                        self.update_idletasks() # update UI (immediately change)
                    print(f"จำนวนไฟล์สมบูรณ์: {normal_file} ภาพ \nจำนวนไฟล์เสียหาย: {corrupted_file} ภาพ") # show result after finish process each folder
                    # if have corrupt file show path for error report ----------
                    if(corrupted_file>0):
                        print(f"error report: {report_file}")
            # ------ if program run till finish ------------------------
            if not(event_obj.is_set()):
                self.run_button.config(text="เสร็จสิ้น", bg="lightgreen", state="disabled") # change run button text
                self.switch_button.config(state=tk.NORMAL) # enable switch page button
                self.dropdown.config(state="readonly") # enable dropdown config button
                self.update_idletasks()  # change UI (immediately change)
                self.hide_button(self.cancle_button) # hide cancel button
                self.update_idletasks()  # change UI (immediately change)
        except np.core._exceptions._ArrayMemoryError as e:
            print("Memory error occurred:", e)
            # Must call GUI code from main thread
            self.after(0, lambda: self.confirm_and_exit("Memory Error", "Memory allocation failed"))
        except Exception as e:
            error_msg = e
            print("Error:", error_msg)
            self.after(0, lambda: self.confirm_and_exit("Error", error_msg))
               
