import json
import pytest
import sys
import glob
import os
import numpy as np
import shutil
import datetime

from . import get_file_path, find_file_path
sys.path.append("..")
from tools.overlay import OverlayGAN, OverlayCube
from tools.utils import find_fol, create_fol, check_corrupt
from page.main_page import load_model_generator


@pytest.fixture(scope="module")
def configs():
    config_path = find_file_path("config_gan_ui.json")
    with open(config_path, "r") as f:
        return json.load(f)
    
@pytest.fixture
def kmutnb_config(configs):
    return configs['kmutnb']

@pytest.fixture
def hii_config(configs):
    return configs['hii']
    
@pytest.fixture
def generator_model(kmutnb_config):
    return load_model_generator(kmutnb_config['model_type'],kmutnb_config["model_input"],kmutnb_config["model_weight"])

@pytest.fixture
def resize_scale():
    return 'resize'

@pytest.fixture
def pad_image_scale():
    return 'pad_image'
    
@pytest.fixture
def kmutnb_input_value():
    return 'example_data/Kmutnb/'

@pytest.fixture
def hii_input_value():
    return 'example_data/Hii/'

@pytest.fixture
def hii_input_value_corrupt():
    return 'example_data/Hii/1_corrupted/'

@pytest.fixture
def hii_input_value_complete():
    return 'example_data/Hii/2_complete/'

@pytest.fixture
def hii_input_value_mixed():
    return 'example_data/Hii/3_mixed/'

@pytest.fixture
def hii_input_value_empty():
    return 'example_data/Hii/4_empty/'

@pytest.fixture
def overlay_kmutnb(configs,resize_scale,generator_model):
    kmutnb_config = configs['kmutnb']
    kmutnb = OverlayGAN(kmutnb_config,resize_scale)
    kmutnb.set_useMask()
    kmutnb.set_model_generator(generator_model)
    return kmutnb

@pytest.fixture
def overlay_kmutnb_back(configs, resize_scale, generator_model):
    kmutnb_config = configs['kmutnb']
    kmutnb = OverlayCube(kmutnb_config,resize_scale)
    kmutnb.set_useMask()
    kmutnb.set_model_generator(generator_model)
    return kmutnb

@pytest.fixture
def overlay_hii(configs,resize_scale,generator_model):
    hii_config = configs['hii']
    hii = OverlayGAN(hii_config,resize_scale)
    hii.set_useMask()
    hii.set_model_generator(generator_model)
    return hii

@pytest.mark.parametrize("test_input, expected_output", [('example_data/Kmutnb/', 1),
                                                         ('example_data/Hii/',3),
                                                         ('example_data/Hii/1_corrupted/', 1), 
                                                         ('example_data/Hii/2_complete/', 1), 
                                                         ('example_data/Hii/3_mixed/', 1),
                                                         ('example_data/Hii/4_empty/', 0)])
def test_find_fol(test_input, expected_output):
    path_input  = find_fol(get_file_path(test_input))
    assert len(path_input) == expected_output

@pytest.mark.parametrize("test_input, expected_fol, expected_img, expected_corrupt", [('example_data/Hii/',3,[1,1,2],[[True],[False],[False,True]]),
                                                         ('example_data/Hii/1_corrupted/', 1,[1],[[True]]), 
                                                         ('example_data/Hii/2_complete/', 1,[1],[[False]]), 
                                                         ('example_data/Hii/3_mixed/', 1,[2],[[False,True]]),
                                                         ('example_data/Hii/4_empty/', 0,[],[[]])])
def test_overlay_top_hii(overlay_hii,test_input,expected_fol,expected_img,expected_corrupt):
    assert overlay_hii.get_haveMask_status() == True
    path_input  = find_fol(get_file_path(test_input))
    path_input.sort()
    print(path_input)
    assert len(path_input) == expected_fol
    if(len(path_input) > 0):
        for i in range(len(path_input)):
            target = []
            main_path = (path_input[i]).replace('\\','/')
            dir_list_input = glob.glob(main_path+'*'+'.jpg')
            dir_list_input.sort()
            assert len(dir_list_input) == expected_img[i]
            line_name = os.path.basename(os.path.dirname(os.path.dirname(main_path)))
            assert line_name == (main_path.split('/'))[-3]
            save_path = (main_path + line_name + '_output').replace('\\','/')
            assert save_path.find((main_path.split('/'))[-3]+'_output') != -1
            time_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            report_file = save_path+'/'+line_name+'_error_report_'+time_now+'.txt'
            if(os.path.exists(save_path)):
                shutil.rmtree(get_file_path(save_path))
            create_fol(save_path)
            assert os.path.exists(save_path) == True
            for j in range(len(dir_list_input)):
                name_file = ((dir_list_input[j].replace('\\','/')).split('/'))[-1]
                num_file = ((name_file.split('_'))[1])[:-4]
                target.append(int(num_file))
            assert len(target) == len(dir_list_input)
            for count_pic, coor in enumerate(target):
                print(expected_corrupt)
                assert check_corrupt(main_path+ "img_"+str(coor).rjust(6, '0')+".jpg",'') == expected_corrupt[i][count_pic]
                if(not check_corrupt(main_path+ "img_"+str(coor).rjust(6, '0')+".jpg",report_file)):
                    inpaint_result = overlay_hii.overlay(main_path,coor)
                    assert isinstance(inpaint_result, np.ndarray)
                    assert inpaint_result.shape == (4096, 8192, 3)
                    overlay_hii.save_pano(inpaint_result,save_path,coor)
                    assert os.path.exists(save_path+ '/'+ "img_"+str(coor).rjust(6, '0')+"_O.jpg")
                else:
                    assert os.path.exists(report_file)


# @pytest.mark.parametrize("test_input, expected_output", [("2+2", 4), ("2-2", 0), ("2*2", 4), ("2/2", 1.0)])
def test_overlay_top_kmutnb(overlay_kmutnb, kmutnb_input_value):
    target = []
    assert overlay_kmutnb.get_haveMask_status() == True
    path_input  = find_fol(get_file_path(kmutnb_input_value))
    assert len(path_input) == 1
    dir_list_input = glob.glob(path_input[0]+'*'+'.jpg')
    assert len(dir_list_input) == 2
    line_name = os.path.basename(os.path.dirname(os.path.dirname(path_input[0])))
    assert line_name == 'Kmutnb'
    save_path = (path_input[0] + line_name + '_output').replace('\\','/')
    assert save_path.find('Kmutnb_output') != -1
    if(os.path.exists(save_path)):
        shutil.rmtree(get_file_path(save_path))
    create_fol(save_path)
    assert os.path.exists(save_path) == True
    for i in range(len(dir_list_input)):
        name_file = (dir_list_input[i].split('\\'))[-1]
        num_file = ((name_file.split('_'))[1])[:-4]
        target.append(int(num_file))
    assert len(target) == 2
    inpaint_result = overlay_kmutnb.overlay(path_input[0],target[0])
    assert isinstance(inpaint_result, np.ndarray)
    assert inpaint_result.shape == (4096, 8192, 3)
    overlay_kmutnb.save_pano(inpaint_result,save_path,target[0])
    assert os.path.exists(save_path+ '/'+ "img_"+str(target[0]).rjust(6, '0')+"_O.jpg")

def test_overlay_back_kmutnb(overlay_kmutnb_back, kmutnb_input_value):
    target = []
    assert overlay_kmutnb_back.get_haveMask_status()['cube_back'] == True
    path_input  = find_fol(get_file_path(kmutnb_input_value))
    assert len(path_input) == 1
    dir_list_input = glob.glob(path_input[0]+'*'+'.jpg')
    assert len(dir_list_input) == 2
    line_name = os.path.basename(os.path.dirname(os.path.dirname(path_input[0])))
    assert line_name == 'Kmutnb'
    save_path = (path_input[0] + line_name + '_output').replace('\\','/')
    assert save_path.find('Kmutnb_output') != -1
    if(os.path.exists(save_path)):
        shutil.rmtree(get_file_path(save_path))
    create_fol(save_path)
    assert os.path.exists(save_path) == True
    for i in range(len(dir_list_input)):
        name_file = (dir_list_input[i].split('\\'))[-1]
        num_file = ((name_file.split('_'))[1])[:-4]
        target.append(int(num_file))
    assert len(target) == 2
    inpaint_result = overlay_kmutnb_back.overlay(path_input[0],target[0])
    assert isinstance(inpaint_result, np.ndarray)
    assert inpaint_result.shape == (4096, 8192, 3)
    overlay_kmutnb_back.save_pano(inpaint_result,save_path,target[0])
    assert os.path.exists(save_path+ '/'+ "img_"+str(target[0]).rjust(6, '0')+"_O.jpg")

def test_overlay_top_back(overlay_kmutnb, overlay_kmutnb_back, kmutnb_input_value):
    target = []
    assert overlay_kmutnb_back.get_haveMask_status()['cube_back'] == True
    path_input  = find_fol(get_file_path(kmutnb_input_value))
    assert len(path_input) == 1
    dir_list_input = glob.glob(path_input[0]+'*'+'.jpg')
    assert len(dir_list_input) == 2
    line_name = os.path.basename(os.path.dirname(os.path.dirname(path_input[0])))
    assert line_name == 'Kmutnb'
    save_path = (path_input[0] + line_name + '_output').replace('\\','/')
    assert save_path.find('Kmutnb_output') != -1
    if(os.path.exists(save_path)):
        shutil.rmtree(get_file_path(save_path))
    create_fol(save_path)
    assert os.path.exists(save_path) == True
    for i in range(len(dir_list_input)):
        name_file = (dir_list_input[i].split('\\'))[-1]
        num_file = ((name_file.split('_'))[1])[:-4]
        target.append(int(num_file))
    assert len(target) == 2
    top_result = overlay_kmutnb.overlay(path_input[0],target[0])
    inpaint_result = overlay_kmutnb_back.overlay(top_result,target[0])
    assert isinstance(inpaint_result, np.ndarray)
    assert inpaint_result.shape == (4096, 8192, 3)
    overlay_kmutnb_back.save_pano(inpaint_result,save_path,target[0])
    assert os.path.exists(save_path+ '/'+ "img_"+str(target[0]).rjust(6, '0')+"_O.jpg")
