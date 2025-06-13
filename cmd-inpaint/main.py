import json

from page.open_page import MainApp
from tools.utils import get_file_path

# Run the Application
if __name__ == "__main__":
    # Load configs
    config_path = get_file_path("config_gan_ui.json")
    with open(config_path, "r") as f:
        configs = json.load(f)
    # print(configs["kmutnb"])
    # initial_config = configs["kmutnb"]
    app = MainApp(configs)
    # app = ImageProcessingApp(configs)
    app.mainloop()