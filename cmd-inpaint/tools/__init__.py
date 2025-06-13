import os
import sys

from .utils import get_file_path

# Project root relative to the tools directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def find_file_path(filename, must_exist=True):
    """
    Locate file from project root, falling back to PyInstaller _MEIPASS.

    Args:
        filename (str): File to find relative to project root.
        must_exist (bool): Raise error if not found.

    Returns:
        str: Path to file.
    """
    return get_file_path(filename, base_path=BASE_DIR, must_exist=must_exist)