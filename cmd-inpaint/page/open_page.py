import tkinter as tk
import sys

from .main_page import ImageProcessingApp
from .mask_page import MaskApp

from . import get_file_path, find_file_path

sys.path.append("..")
# from tools.utils import get_file_path


# ------------ main windows -------------------------------------

class MainApp(tk.Tk):
    def __init__(self,configs):
        super().__init__()
        self.title("Inpaint System Ver.1.4")
        self.geometry("700x550")
        icon_path = "image_app/icon.ico"
        self.iconbitmap(find_file_path(icon_path))

        # container holds all pages
        self.container = tk.Frame(self)
        self.container.pack()

        # print(configs.get(list(configs.keys())[0], {}))

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
            "config_name": list(configs.keys())[0],
            "config_data": configs.get(list(configs.keys())[0], {}),
            "segment_model": None,
            "type_process_scale": 'resize',
            "mask_edited": False
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
        self.build_menu_for_frame(frame)


        # Dynamically resize the window
        if page_class == 'ImageProcessingApp':
            self.geometry("615x560")
        elif page_class == 'MaskApp':
            self.geometry("1200x650")

        # Call on_show() if it exists
        if hasattr(frame, "on_show"):
            frame.start_thread_on_show()

    def build_menu_for_frame(self, frame):
        menu = tk.Menu(self)

        if isinstance(frame, MaskApp):
            files_menu = tk.Menu(menu, tearoff=0) # tearoff - remove dashed line before menu item
            files_menu.add_command(label="Load new image", command=frame.load_mask)
            files_menu.add_command(label="Load config setting", command=frame.reload_config)
            # files_menu.add_separator()
            # files_menu.add_command(label="Back", command=frame.back_step)
            menu.add_cascade(label="Files", menu=files_menu)

            tools_menu = tk.Menu(menu, tearoff=0)
            tools_menu.add_command(label="Rectangle mask", command=frame.direct_draw)
            tools_menu.add_command(label="Segment mask", command=frame.start_thread_run_segment)
            tools_menu.add_command(label="Clear mask", command=frame.clear_mask)
            tools_menu.add_command(label="Save mask", command=frame.save_mask)
            menu.add_cascade(label="Tools", menu=tools_menu)

            edits_menu = tk.Menu(menu, tearoff=0)
            edits_menu.add_command(label="New config", command=frame.new_config)
            edits_menu.add_command(label="Save config", command=frame.save_config)
            menu.add_cascade(label="Edits config", menu=edits_menu)

            menu.add_command(label="Next step", command=frame.next_step)


        self.config(menu=menu)