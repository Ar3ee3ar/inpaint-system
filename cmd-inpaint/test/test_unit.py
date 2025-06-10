import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
import sys
import os
import json

# Add parent directory to sys.path
sys.path.append("..")
print('first: ',os.getcwd())
from gen_gan_img_ui_deepfill import MainApp,ImageProcessingApp, get_file_path

class TestTkinterUI(unittest.TestCase):
    def setUp(self):
        # Set up a root window for each test
        config_path = get_file_path("config_gan_ui.json")
        with open(config_path, "r") as f:
            configs = json.load(f)
        self.configs = configs
        self.root = MainApp(configs)
        self.root.withdraw()  # Hide the main window
        self.config = configs[list(configs.keys())[0]]
        print(self.config)

    def tearDown(self):
        self.root.destroy()

    def test_ui_elements_exist(self):
        # Assuming your UI class is called App or similar
        app = ImageProcessingApp(self.root.container, self.root, self.configs)
        # Check if a button exists
        # self.assertIsInstance(app.run_button, tk.Button)
        # # Check if an entry exists
        self.assertIsInstance(app.dir_entry, tk.Entry)

    # @patch('gen_gan_img_ui_deepfill.some_processing_function')
    # def test_button_triggers_processing(self, mock_processing):
    #     app = gen_gan_img_ui_deepfill.App(self.root)
    #     app.input_entry.insert(0, 'test.png')
    #     app.mask_entry.insert(0, 'mask.png')
    #     app.output_entry.insert(0, 'output.png')
    #     # Simulate button click
    #     app.run_button.invoke()
    #     mock_processing.assert_called_once()

if __name__ == '__main__':
    unittest.main()
