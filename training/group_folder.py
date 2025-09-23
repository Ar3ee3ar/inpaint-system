import argparse
from pathlib import Path
from tqdm import tqdm
import os
from distutils.util import strtobool

# Description: command ไว้ทำตามคำสั่ง
def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str, default='', help="set format for different type of model df|p2p dataset")
    parser.add_argument("--config",type=str, default='', help="config dataset name")
    parser.add_argument("--dest_fol", type=str, default='', help="final folder for stored data")
    parser.add_argument("--isCombine", type=str, nargs='?', const='True', default="False", help="if dataset come from more than 1 dataset. Can notice from format of image name e.g. dataset-img_000000.jpg")
    arg = parser.parse_args()
    return arg

def main():
    type_folder = ["train","test","val"]
    format_mode = _argparse().mode
    config_name = _argparse().config
    combine_flag = bool(strtobool(_argparse().isCombine))
    dest_fol = _argparse().dest_fol
    dest_fol_path = f"car_ds/pic/{dest_fol}"
    train_config_path = f"car_ds/train_test_config/{config_name}/"
    if not os.path.exists(dest_fol_path): 
        os.makedirs(dest_fol_path)
    for fol in type_folder:
        if(format_mode == "df"):
            # path = "D:/inpaint_gan/car_ds/pic/"+config_name+"/"+fol
            path = os.path.join(Path(__file__).parent.absolute(), "car_ds/pic/"+dest_fol+"/"+fol)
            current_diectory = str(Path(__file__).parent.absolute()).replace('\\','/') + '/'
            if not os.path.exists(path): 
                os.makedirs(path)
            with open(train_config_path+fol+"/input.txt") as f:
                for count, line in enumerate(tqdm(f)):
                    get_file_name = (line.strip()).split('/')
                    if(combine_flag):
                        img_path = current_diectory+get_file_name[0]+"/"+get_file_name[1]+"/"+dest_fol+"/"+fol+"/"+get_file_name[2]+'-'+get_file_name[3]
                    else:
                        img_path = current_diectory+get_file_name[0]+"/"+get_file_name[1]+"/"+dest_fol+"/"+fol+"/"+get_file_name[3]
                    try:
                        Path(current_diectory+line.strip()).rename(img_path)
                    except FileNotFoundError:
                        # pass
                        print(f"File not found: {current_diectory+line.strip()}")
            with open(train_config_path+fol+"/masks.txt") as f:
                for count, line in enumerate(tqdm(f)):
                    get_file_name = (line.strip()).split('/')
                    if(combine_flag):
                        img_path = current_diectory+get_file_name[0]+"/"+get_file_name[1]+"/"+dest_fol+"/"+fol+"/"+get_file_name[2]+'-'+get_file_name[3]
                    else:
                        img_path = current_diectory+get_file_name[0]+"/"+get_file_name[1]+"/"+dest_fol+"/"+fol+"/"+get_file_name[3]
                    try:
                        Path(current_diectory+line.strip()).rename(img_path)
                    except FileNotFoundError:
                        # pass
                        print(f"File not found: {current_diectory+line.strip()}")
        elif(format_mode == "p2p"):
            current_diectory = str(Path(__file__).parent.absolute()).replace('\\','/') + '/'
            with open(train_config_path+fol+"/input.txt") as f:
                for count, line in enumerate(tqdm(f)):
                    get_file_name = (line.strip()).split('/')
                    if(combine_flag):
                        img_path = current_diectory+get_file_name[0]+"/"+get_file_name[1]+"/"+dest_fol+"/"+fol+"/"+get_file_name[2]+'-'+get_file_name[3]
                    else:
                        img_path = current_diectory+get_file_name[0]+"/"+get_file_name[1]+"/"+dest_fol+"/"+fol+"/"+get_file_name[3]
                    try:
                        Path(img_path).rename(current_diectory+line.strip())
                    except FileNotFoundError:
                        print(f'file not found: {img_path}')
                        # pass
                        # print(f"File not found: {current_diectory
                                # +get_file_name[0]+"/"
                                # +get_file_name[1]+"/"
                                # +dest_fol+"/"
                                # +fol+"/"+config_name+'-'+get_file_name[3]}")
            with open(train_config_path+fol+"/masks.txt") as f:
                for count, line in enumerate(tqdm(f)):
                    get_file_name = (line.strip()).split('/')
                    if(combine_flag):
                        img_path = current_diectory+get_file_name[0]+"/"+get_file_name[1]+"/"+dest_fol+"/"+fol+"/"+get_file_name[2]+'-'+get_file_name[3]
                    else:
                        img_path = current_diectory+get_file_name[0]+"/"+get_file_name[1]+"/"+dest_fol+"/"+fol+"/"+get_file_name[3]
                    try:
                        Path(img_path).rename(current_diectory+line.strip())
                    except FileNotFoundError:
                        print(f'file not found: {img_path}')
                        # pass
                        # print(f"File not found: {current_diectory
                                # +get_file_name[0]+"/"
                                # +get_file_name[1]+"/"
                                # +dest_fol+"/"
                                # +fol+"/"+config_name+'-'+get_file_name[3]}")

if __name__ == '__main__':
    main()
