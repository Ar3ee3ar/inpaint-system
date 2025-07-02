import tkinter as tk
import sys

from .main_page import ImageProcessingApp
from .mask_page import MaskApp

from . import get_file_path, find_file_path

# ------------ main windows -------------------------------------
class MainApp(tk.Tk):
    """
    Main application class for the Inpaint System GUI.
    Inherits from:
        tk.Tk: The main window object from Tkinter.
    Args:
        configs (dict): A dictionary containing configuration data for the application.
    Attributes:
        container (tk.Frame): The main container frame that holds all pages.
        shared_data (dict): A dictionary for sharing data between different pages of the app.
        frames (dict): A dictionary mapping page names to their corresponding frame instances.
    Methods:
        show_frame(page_class):
            Displays the specified page/frame, resizes the window accordingly, and calls the frame's on_show method if it exists.
        build_menu_for_frame(frame):
            Builds and sets the menu bar for the given frame, adding context-specific menu options.
    """

    def __init__(self, configs:dict):
        super().__init__()
        self.title("Inpaint System Ver.1.4")
        self.geometry("700x550")
        icon_path = "image_app/icon.ico"
        self.iconbitmap(find_file_path(icon_path))

        # container holds all pages
        self.container = tk.Frame(self)
        self.container.pack()

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

    def show_frame(self, page_class:str):
        """
        Displays the specified frame (page) in the application, raises it to the top,
        builds its associated menu, and dynamically resizes the window based on the page.
        If the frame has an 'on_show' method, starts its associated thread.
        Args:
            page_class (str): The key identifying the frame to display.
        Raises:
            KeyError: If the specified page_class does not exist in self.frames.
        """
        frame = self.frames[page_class]
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()
        self.build_menu_for_frame(frame)


        # Dynamically resize the window
        if page_class == 'ImageProcessingApp':
            self.geometry("615x560")
        elif page_class == 'MaskApp':
            self.geometry("1200x650")
        else:
            raise KeyError(f"Wrong page_class: '{page_class}' not found in frames.")

        # Call on_show() if it exists
        if hasattr(frame, "on_show"):
            frame.start_thread_on_show()

    def build_menu_for_frame(self, frame:tk.Frame) -> None:
        """
        Builds and sets the menu bar for the given frame in the application.
        Depending on the type of the provided frame, this method creates and attaches
        a menu bar with various options. If the frame is an instance of MaskApp, it
        adds "Files", "Tools", and "Edits config" menus with commands for loading images,
        managing masks, and handling configuration settings. It also adds a "Next step"
        command to the menu bar.
        Parameters:
            frame (tk.Frame): The frame for which the menu should be built. If the frame
                              is an instance of MaskApp, specific menu options are added.
        Returns:
            None
        """

        menu = tk.Menu(self)

        if isinstance(frame, MaskApp):
            files_menu = tk.Menu(menu, tearoff=0) # tearoff - remove dashed line before menu item
            files_menu.add_command(label="Load new image", command=frame.load_mask) # for load new mask
            files_menu.add_command(label="Load config setting", command=frame.reload_config) # for reload config setting mask
            # files_menu.add_separator()
            # files_menu.add_command(label="Back", command=frame.back_step)
            menu.add_cascade(label="Files", menu=files_menu) # add Files to main menu for this cascade

            tools_menu = tk.Menu(menu, tearoff=0)
            tools_menu.add_command(label="Rectangle mask", command=frame.direct_draw) # for create rectangle mask by xyxy box that user created
            tools_menu.add_command(label="Segment mask", command=frame.start_thread_run_segment) # for created segment mask from SAM model by get input from user xyxy box
            tools_menu.add_command(label="Clear mask", command=frame.clear_mask) # remove mask
            tools_menu.add_command(label="Save mask", command=frame.save_mask) # save mask as file image
            menu.add_cascade(label="Tools", menu=tools_menu) # add Files to main menu for this cascade

            edits_menu = tk.Menu(menu, tearoff=0)
            edits_menu.add_command(label="New config", command=frame.new_config) # add new config setting by save image and variable to json
            edits_menu.add_command(label="Save config", command=frame.save_config) # save new mask to current selected config setting
            menu.add_cascade(label="Edits config", menu=edits_menu) # add Edits config to main mwnu for this cascade

            menu.add_command(label="Next step", command=frame.next_step) # next step button for pass mask setting to main_page
        # elif isinstance(frame,ImageProcessingApp):


        self.config(menu=menu)