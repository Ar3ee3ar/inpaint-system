import tkinter as tk
from tkinter import ttk
import threading
import time

class LoadingPopup(tk.Toplevel):
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
        self.progress.stop()
        self.destroy()