import json
import os
import sys
import shutil

from page.open_page import MainApp
# from tools.utils import get_file_path

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

# Run the Application
if __name__ == "__main__":
    # if hasattr(sys, '_MEIPASS'):
    #     print("PyInstaller _MEIPASS temp folder:")
    #     print(sys._MEIPASS)
    #     temp_dir = sys._MEIPASS
    #     shutil.copytree(temp_dir, "meipass_copy", dirs_exist_ok=True)
    #     print("Copied to ./meipass_copy")
    # Load configs
    config_path = get_file_path("config_gan_ui.json")
    with open(config_path, "r") as f:
        configs = json.load(f)
    # print(configs["kmutnb"])
    # initial_config = configs["kmutnb"]
    app = MainApp(configs)
    # app = ImageProcessingApp(configs)
    app.mainloop()