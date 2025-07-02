import tkinter as tk
from tkinter import ttk
import threading
import time

class LoadingPopup(tk.Toplevel):
    """
    A modal popup window displaying a loading message and an indeterminate progress bar.
    Args:
        parent (tk.Widget): The parent window.
        message (str, optional): The message to display above the progress bar. Defaults to "Please wait...".
    Methods:
        close():
            Stops the progress bar and closes the popup window.
    Notes:
        - The popup is modal and cannot be closed by the user.
        - Intended to be used as a temporary indicator during long-running operations.
    """

    def __init__(self, parent, message="Please wait..."):
        super().__init__(parent)
        self.title("Loading")
        self.geometry("250x100")
        self.resizable(False, False)

        tk.Label(self, text=message).pack(pady=10)
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.pack(fill='x', padx=20, pady=10)
        self.progress.start(10)

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable closing

    def close(self):
        """
        Stops the progress bar and closes the popup window.
        """
        self.progress.stop()
        self.destroy()