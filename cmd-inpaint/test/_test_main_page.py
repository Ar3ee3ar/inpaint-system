import pytest
import tkinter as tk
import json
import sys
import os

from . import get_file_path, find_file_path
# Add parent directory to sys.path
sys.path.append("..")
from page.open_page import MainApp
from page.main_page import ImageProcessingApp

@pytest.fixture(scope="module")
def configs():
    config_path = find_file_path("config_gan_ui.json")
    with open(config_path, "r") as f:
        return json.load(f)

@pytest.fixture
def main_app(configs):
    app = MainApp(configs)
    # app.mainloop()
    app.withdraw()
    yield app
    app.destroy()

def test_get_file_path_returns_existing_file():
    path = find_file_path("config_gan_ui.json")
    path_icon = find_file_path('image_app/icon.ico')
    path_process = find_file_path("image_app/flow.png")
    assert os.path.exists(path)
    assert os.path.exists(path_icon)
    assert os.path.exists(path_process)

def test_main_app_initialization(main_app):
    assert isinstance(main_app, MainApp)
    assert hasattr(main_app, "container")

@pytest.fixture
def image_processing_app(main_app, configs):
    config = configs[list(configs.keys())[0]]
    app = ImageProcessingApp(main_app.container, main_app, configs)
    yield app

def test_image_processing_app_elements(image_processing_app):
    assert isinstance(image_processing_app.dir_entry, tk.Entry)
    assert hasattr(image_processing_app, "run_button")
    assert isinstance(image_processing_app.run_button, tk.Button)

def test_image_processing_app_config(image_processing_app, configs):
    # Check if the app has access to configs
    assert image_processing_app.configs == configs

def test_dir_entry_default_value(image_processing_app):
    # Check if dir_entry is empty by default
    assert image_processing_app.dir_entry.get() == ""

def test_run_button_command(image_processing_app, monkeypatch):
    called = {}

    def fake_run():
        called["run"] = True

    image_processing_app.run_button.config(command=fake_run)
    image_processing_app.run_button.invoke()
    assert called.get("run") is True

