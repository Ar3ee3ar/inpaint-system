import os
import sys

def get_file_path(filename):
    """
    Try to locate the file either in the bundled PyInstaller directory (_MEIPASS)
    or externally next to the executable or script.
    """
    paths_to_try = []

    # If running from PyInstaller
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            paths_to_try.append(os.path.join(sys._MEIPASS, filename))  # inside EXE
        paths_to_try.append(os.path.join(os.path.dirname(sys.executable), filename))  # next to EXE
    else:
        # fallback for source mode
        base_path = os.path.dirname(os.path.abspath(__file__))
        paths_to_try.append(os.path.join(base_path, filename))

    for path in paths_to_try:
        if os.path.exists(path):
            return path

    return ""

def find_file_path(filename):
    """
    Locate file relative to the project root, compatible with PyInstaller.
    """
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(current_file, "..", ".."))  # Adjust as needed
    rel_path = os.path.normpath(os.path.join(project_root, filename))

    if getattr(sys, 'frozen', False):
        # First try PyInstaller paths
        result_path = get_file_path(filename)
        if result_path == "":
            # Then try project root fallback
            result_path = get_file_path(rel_path)
        return result_path

    # Normal source mode: just use full relative path from root
    return get_file_path(rel_path)

