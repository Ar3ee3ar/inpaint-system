import argparse
import cv2
import numpy as np
import os
import sys
import time
from PIL import Image
import time
import utm
import scipy.sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import glob
from distutils.util import strtobool
import json

# sys.path.append("..")

from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from tools.process_pano import Equirectangular,Perspective,inpaint_pano
# from initial_gps import *

# degree สะพาน (167, 0, -90, 1000, 1000)
# degree สะพาน (167, 27, -80, 1000, 1000)

def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-main_dir",type=str, default='', help="parent directory that store dataset")
    parser.add_argument("--overwrite", "-overwrite", type=str, nargs='?', const='True', default="False", help="enable wandb log")
    parser.add_argument("--config", "-config",type=str, default='', help="config cars")
    arg = parser.parse_args()
    return arg

# Helper to get file path near the .exe
def get_file_path(filename):
    """
    Try to locate the file either in the bundled PyInstaller directory (_MEIPASS)
    or externally next to the executable or script.
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

def create_fol(path):
    if not os.path.exists(path): 
        os.makedirs(path)
    else:
        print('already create '+path)

def find_fol(main_input):
    dir_list_input = glob.glob(main_input+'/**/'+'Camera/',recursive=True)
    for i in range(len(dir_list_input)):
        dir_list_input[i] = dir_list_input[i].replace('\\','/')
    return dir_list_input

class OverlayGPS():
    def __init__(self, config_data, config_name):
        print('Start read '+config_name+' config...')
        self.coor = config_data["coor"]
        self.mask = config_data["mask"]
        self.img_top_h = config_data["img_top_h"]
        self.img_top_w = config_data["img_top_w"]
        self.img_pano_h =  config_data["img_pano_h"]
        self.img_pano_w = config_data["img_pano_w"]
        self.seamless_center_h = config_data["seamless_center_h"]
        self.seamless_center_w = config_data["seamless_center_w"]
        self.move_img_up = config_data["move_img_up"]
        self.move_img_down = config_data["move_img_down"]
        self.x_new_coor = config_data["x_new_coor"]
        self.y_new_coor = config_data["y_new_coor"]
        self.relate_dist = config_data["relate_dist"]

        self.mask_img = cv2.resize(cv2.imread(get_file_path(self.mask)),(self.img_top_h,self.img_top_w)) # for model
        # project to equirectangular (get mask image)
        equ = Equirectangular(self.mask_img, 167, self.coor, -90,'img')    
        mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w) 
        self.mask_pano_img = mask_pano_img.astype(np.uint8)
        print('Complete setting '+config_name)

    def laplacian_matrix(self,n, m):
        """Generate the Poisson matrix. 

        Refer to: 
        https://en.wikipedia.org/wiki/Discrete_Poisson_equation

        Note: it's the transpose of the wiki's matrix 
        """
        mat_D = scipy.sparse.lil_matrix((m, m))
        mat_D.setdiag(-1, -1)
        mat_D.setdiag(4)
        mat_D.setdiag(-1, 1)
        mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
        mat_A.setdiag(-1, 1*m)
        mat_A.setdiag(-1, -1*m)
        return mat_A


    def poisson_edit(self,source, target, mask, offset):
        """The poisson blending function. 

        Refer to: 
        Perez et. al., "Poisson Image Editing", 2003.
        """
        y_max, x_max = target.shape[:-1]
        y_min, x_min = 0, 0
        x_range = x_max - x_min
        y_range = y_max - y_min
        M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
        source = cv2.warpAffine(source,M,(x_range,y_range))
        mask = mask[y_min:y_max, x_min:x_max]    
        mask[mask != 0] = 1
        mat_A = self.laplacian_matrix(y_range, x_range)
        laplacian = mat_A.tocsc() 
        for y in range(1, y_range - 1):
            for x in range(1, x_range - 1):
                if mask[y, x] == 0:
                    k = x + y * x_range
                    mat_A[k, k] = 1
                    mat_A[k, k + 1] = 0
                    mat_A[k, k - 1] = 0
                    mat_A[k, k + x_range] = 0
                    mat_A[k, k - x_range] = 0
        mat_A = mat_A.tocsc()
        mask_flat = mask.flatten()    
        for channel in range(source.shape[2]):
            source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
            target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()              
            alpha = 1
            mat_b = laplacian.dot(source_flat)*alpha
            mat_b[mask_flat==0] = target_flat[mask_flat==0]
            x = spsolve(mat_A, mat_b)
            x = x.reshape((y_range, x_range))
            x[x > 255] = 255
            x[x < 0] = 0
            x = x.astype('uint8')
            target[y_min:y_max, x_min:x_max, channel] = x

        return target

    def seamless(self,source,target):
        mask = self.mask_img
        (h, w) = (self.seamless_center_h, self.seamless_center_w)
        center = (h,w)
        # center = (1002,901)
        im_clone = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
        return im_clone

    def convert_yaw_to_360(self,yaw):
        yaw_360 = (yaw + 360) % 360
        return yaw_360

    def find_relate_new(self,coor_detail,target_pic, type_coor):
        qt = self.relate_dist
        if(type_coor == "latlon"):
            target_coor = self.latlon_to_utm([coor_detail["x"][target_pic], coor_detail["y"][target_pic]])
        else:
            target_coor = [coor_detail["x"][target_pic], coor_detail["y"][target_pic]]
        xt, yt = target_coor[0:2]
        min_x_dist = 9999999
        min_y_dist = 9999999
        sum_dist = 9999999
        min_pic = target_pic
        # for i in range(0,len(coor_detail["x"])):
        for i in range(0,target_pic):
            for range_qt in qt:
                theta = coor_detail["yaw"][i]
                xq = range_qt*np.sin(np.radians(theta)) + xt
                yq = range_qt*np.cos(np.radians(theta)) + yt
                x_dist = np.abs(xq - coor_detail["x"][i])
                y_dist = np.abs(yq - coor_detail["y"][i])
                if((x_dist + y_dist < sum_dist)):
                    if((x_dist< min_x_dist) or (y_dist< min_y_dist)):
                        if((np.abs(coor_detail["x"][target_pic] - coor_detail["x"][i]) >= 8.8) or (np.abs(coor_detail["y"][target_pic] - coor_detail["y"][i]) >= 8.8)):
                            if((np.abs(coor_detail["z"][target_pic] - coor_detail["z"][i]) <= 4)):
                                min_x_dist = x_dist
                                min_y_dist = y_dist
                                min_pic = i
                                sum_dist = min_x_dist + min_y_dist
        return min_pic

    # def save_per(self,img_path,save_path,count):
    #     car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
    #     equ = Perspective(car_pano_img) 
    #     per_img_1000 = equ.GetPerspective(167, self.coor, -90, 1500, 1500)
    #     input_image_im = Image.fromarray((per_img_1000).astype(np.uint8))
    #     input_image_im.save(save_path+count+"_topview_source.jpg")
    

    def equi_to_per(self,img_path,name='',save_path=''):
        car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
        equ = Perspective(car_pano_img)
        per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)
        if(name != ''):
            cv2.imwrite(save_path + '/'+ "img_"+str(name).rjust(6, '0')+"_top.jpg", cv2.resize(per_img_1000,(1000,1000)))
        return per_img_1000

    def per_to_equi(self,save_path,car_path,per_img,count):
        car_pano_img = cv2.imread(car_path)

        equ = Equirectangular(per_img,167, self.coor, -90)    
        inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
        inpaint_fill = inpaint_pano(inpaint_pano_img,self.mask_pano_img,car_pano_img)
        inpaint_fill = (inpaint_fill*255.0).astype(np.uint8)
        cv2.imwrite(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg", inpaint_fill)

    def latlon_to_utm(self,coor):
        utm_coor = utm.from_latlon(coor[0],coor[1])
        return utm_coor

    def utm_to_latlon(self,coor):
        latlon_coor = utm.to_latlon(coor[0],coor[1],coor[2],coor[3])
        return latlon_coor

    def rotate(self,image,degree):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -degree, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def crude_inpaint(self,source, target):
        mask = self.mask_img
        reverse_mask = cv2.bitwise_not(mask)
        source = source/255.0
        target = target/255.0
        reverse_mask[reverse_mask != 0] = 1
        result = reverse_mask * target + (1 - reverse_mask) * source
        return result

    def overlay(self,main_path,save_path,source_path, target_path,count,coor_detail):
        # mask_path = "car_ds/pic/mask/8k_focus_simple.jpg"
        # offset = (0,0)
        source = cv2.imread(main_path + "img_"+str(source_path).rjust(6, '0')+".jpg") 
        target = cv2.imread(main_path + "img_"+str(target_path).rjust(6, '0')+".jpg") 
        source = self.equi_to_per(source,source_path,save_path)
        target = self.equi_to_per(target,target_path,save_path)
        dist_utm_x = coor_detail["x"][source_path] - coor_detail["x"][target_path]
        dist_utm_y = coor_detail["y"][source_path] - coor_detail["y"][target_path]
        dist_degree = self.convert_yaw_to_360(coor_detail["yaw"][source_path]) - self.convert_yaw_to_360(coor_detail["yaw"][target_path])
        new_coor =  self.find_new_coor(dist_utm_x,dist_utm_y,dist_degree)
        if(dist_utm_x >= dist_utm_y):
            utm_direction = "E"
        elif(dist_utm_x < dist_utm_y):
            utm_direction = "N"
        else:
            print('wrong compare')

        if((coor_detail["yaw"][source_path] < 0 and coor_detail["yaw"][source_path] >=0) or (coor_detail["yaw"][source_path] > 0 and coor_detail["yaw"][source_path] <=0)):
            dist_reverse = "Y"
        if(dist_utm_x < dist_utm_y):
            dist_reverse = "N"
        else:
            dist_reverse = "N"

        if(source_path > target_path):
            source = self.move(source,"down",dist_utm_y)
        elif(source_path < target_path):
            source = self.move(source,"up",dist_utm_y)
        else:
            source = self.move(source,"up",7)
        shift_coor = self.m_to_px(new_coor[0]-dist_utm_x, new_coor[1]-dist_utm_y)
        source = cv2.warpAffine(source, np.float32([[1, 0, shift_coor[0]], [0, 1, shift_coor[1]]]), (self.img_top_h,self.img_top_w))
        source = self.rotate(source,self.convert_yaw_to_360(coor_detail["yaw"][target_path]) - self.convert_yaw_to_360(coor_detail["yaw"][source_path]))
        result = self.seamless(source,target)
        self.per_to_equi(save_path, main_path + "img_"+str(target_path).rjust(6, '0')+".jpg",result,count)

    def move(self,img,direction,dist):
        if(direction == "up"):
            T = np.float32([[1, 0, 0], [0, 1, self.move_img_up]])
        elif(direction == "down"):
            T = np.float32([[1, 0, 0], [0, 1, self.move_img_down]]) 
        img_translation = cv2.warpAffine(img, T, (self.img_top_h,self.img_top_w))
        return img_translation

    def find_new_coor(self,x,y,degree):
        x_new = x*np.cos(np.radians(degree)) + y*np.sin(np.radians(degree))
        y_new = -x*np.sin(np.radians(degree)) + y*np.cos(np.radians(degree))
        return (x_new,y_new)

    def m_to_px(self,X,Y):
        X_new = ((self.x_new_coor[0]*X) +self.x_new_coor[1])
        Y_new = ((self.y_new_coor[0]*Y) +self.y_new_coor[1])
        return (X_new, Y_new)

    def overlay_process(self,main_path,save_path, coor_detail,overwrite):
        print(main_path)
        target = []
        dir_list_input = glob.glob(main_path+'*'+'.jpg')
        dir_list_input.sort()
        if(not os.path.exists(save_path)):
            create_fol(save_path)
        else:
            if(not bool(strtobool(overwrite))):
                for item in dir_list_input[:]: 
                    pic_full_name = item.split('\\')[-1]
                    pic_name = save_path+'/'+pic_full_name[:-4]+'_O.jpg'
                    if(os.path.exists(pic_name)):
                        dir_list_input.remove(item)
        if(len(dir_list_input)>0):
            for i in range(len(dir_list_input)):
                name_file = (dir_list_input[i].split('\\'))[1]
                num_file = ((name_file.split('_'))[1])[:-4]
                target.append(int(num_file))
        
            for coor in tqdm(target):
                source_pic = self.find_relate_new(coor_detail,coor,"utm")
                self.overlay(main_path,save_path,source_pic,coor,str(coor),coor_detail)

def main():
    # print(_argparse().dir)
    # main_input = "E:/ProcessMMS/" #NOTE: change man path here
    config_path = get_file_path("config_ui.json")
    with open(config_path, "r") as f:
        configs = json.load(f)
    config_name = _argparse().config
    config_data = configs[config_name]
    over_class = OverlayGPS(config_data,config_name)
    main_input = _argparse().dir
    path_input = find_fol(main_input)
    for i in range(len(path_input)):
        line_name = os.path.basename(os.path.dirname(os.path.dirname(path_input[i])))
        save_path = path_input[i]+line_name+'_output'
        # print(save_path)
        dir_list_input = glob.glob(path_input[i]+'*'+'_camera_position.txt')
        if(len(dir_list_input)>0):
            # print(dir_list_input)
            coor_path = dir_list_input[0]
            coor_detail = {
                "x":[],
                "y": [],
                "z":[],
                "roll": [],
                "pitch": [],
                "yaw": []
            }

            with open(coor_path, 'r', encoding='utf-8') as f:
                for count, line in enumerate(f):
                    if(count > 0):
                        # print(line)
                        coor_list = (line.strip()).split(',')
                        coor_detail["x"].append(float(coor_list[3]))
                        coor_detail["y"].append(float(coor_list[4]))
                        coor_detail["z"].append(float(coor_list[5]))
                        coor_detail["roll"].append(float(coor_list[7]))
                        coor_detail["pitch"].append(float(coor_list[8]))
                        coor_detail["yaw"].append(float(coor_list[6]))
                        # print(count)
            # print(coor_detail["x"][0])
            over_class.overlay_process(path_input[i],save_path,coor_detail,_argparse().overwrite)

if __name__ == '__main__':
    main()