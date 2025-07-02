import json
import os
import sys
import shutil

from page.open_page import MainApp

# Helper to get file path near the .exe
def get_file_path(filename:str) -> str:
    """
    Locate a file by searching common locations depending on how the application is run.
    This function attempts to find the specified file by checking:
        1. The PyInstaller temporary directory (_MEIPASS) if running as a bundled executable.
        2. The directory containing the executable if running as a bundled executable.
        3. The directory containing the script if running from source.
    Args:
        filename (str): The name of the file to locate.
    Returns:
        str: The absolute path to the located file.
    Raises:
        FileNotFoundError: If the file cannot be found in any of the searched locations.
    Notes:
        - search for file path that in same hierarchy of caller
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
    # Load configs file
    config_path = 'config_gan_ui.json'
    with open(config_path, "r") as f:
        configs = json.load(f)
    # Call main_page to initialize tkinter gui
    app = MainApp(configs)
    # an infinite loop used to run the application, wait for an event to occur, and process the event as long as the window is not closed.
    app.mainloop()