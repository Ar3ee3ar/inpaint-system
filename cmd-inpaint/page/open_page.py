import tkinter as tk
import sys

from .main_page import ImageProcessingApp
from .mask_page import MaskApp

sys.path.append("..")
from tools.utils import get_file_path


# ------------ main windows -------------------------------------

class MainApp(tk.Tk):
    def __init__(self,configs):
        super().__init__()
        self.title("Inpaint System Ver.1.4")
        self.geometry("700x550")
        icon_path = "image_app/icon.ico"
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
            "config_data": configs.get(list(configs.keys())[0], {}),
            "segment_model": None,
            "type_process_scale": 'resize'
        }

        # store pages in a dict
        self.frames = {}

        for F in (ImageProcessingApp, MaskApp):
            page_name = F.__name__
            if(F == ImageProcessingApp):
                frame = F(parent=self.container, controller = self, configs = configs)
            elif(F == MaskApp):
                frame = F(parent=self.container, controller = self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame('ImageProcessingApp')

    def show_frame(self, page_class):
        frame = self.frames[page_class]
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()


        # Dynamically resize the window
        if page_class == 'ImageProcessingApp':
            self.geometry("615x560")
        elif page_class == 'MaskApp':
            self.geometry("1400x650")

        # Call on_show() if it exists
        if hasattr(frame, "on_show"):
            frame.on_show()