from PIL import Image
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import pathlib
import yaml

from tools.process_img import preprocess_img, move_img, flip_img, zoom_img, crop_center
from tools.process_pano import Equirectangular,Perspective,inpaint_pano
from tools.utils import binary_change, up_downscale, gen_tranparent_mask ,create_fol
from tools.py360convert.py360convert import c2e, e2c, utils

def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-main_dir",type=str, default='', help="parent directory that store dataset")
    parser.add_argument("--conf", "-conf", type=str, default='', help='image config setting')
    parser.add_argument("--save", "-save", type=str, default='', help='save path for stored output image')
    arg = parser.parse_args()
    return arg

# def gen_tranparent_mask(src):
#     # Convert image to image gray 
#     tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 
    
#     # Applying thresholding technique 
#     _, alpha = cv2.threshold(tmp, thresh=127, maxval=255, type=cv2.THRESH_BINARY) 
    
#     # Using cv2.split() to split channels  
#     # of coloured image 
#     b, g, r = cv2.split(src) 
    
#     # Making list of Red, Green, Blue 
#     # Channels and alpha 
#     rgba = [b, g, r, alpha] 
    
#     # Using cv2.merge() to merge rgba 
#     # into a coloured/multi-channeled image 
#     dst = cv2.merge(rgba, 4) 
#     # cv2.imshow('Transparent Mask', dst)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     # Writing and saving to a new image 
#     # cv2.imwrite(save_path+count+"_gfg_white.png", dst) 
#     return dst

# def binary_change(mask,save_path):
#     # Apply threshold to make it binary (0 and 1)
#     mask = cv2.imread(mask, cv2.IMREAD_COLOR)
#     _, binary_mask = cv2.threshold(mask, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
#     trans_mask = gen_tranparent_mask(binary_mask)
#     # print(binary_mask)
#     # plt.imshow(trans_mask)
#     # plt.show()
#     cv2.imwrite(save_path,trans_mask)
#     # return binary_mask
class process_aug:
    def __init__(self, image_path: str, output_path: str, image_config: dict, mask_path: list, image_mode:dict, want_result:dict) -> None:
        self.img_path = image_path
        self.output_path = output_path
        self.image_config = image_config
        # Loop through mask_path and preprocess each mask image
        self.mask_path = [
            preprocess_img(m, img_size=(self.image_config['img_top_w'], self.image_config['img_top_h']))
            for m in mask_path
        ]
        self.len_mask = len(self.mask_path)
        self.image_mode = image_mode
        self.want_result = want_result
        self.current_mode = ''

    def save_image(self, image:np.ndarray, save_path:str) -> None:
        """
        Turn image to opencv color format then save it
        """
        height, width = image.shape[:2]
        # If image is a PIL Image, convert to OpenCV format and BGR color
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if(height > 512 and width > 512):
            cv2.resize(image, self.image_config['model_input'])
        cv2.imwrite(save_path, image)

    def binary_trans_mask(self, img:np.ndarray, save_path:str='') -> np.ndarray:
        """
        Binarized image and turn it into transparent format then save it
        """
        binary_mask = binary_change(img)
        trans_mask = gen_tranparent_mask(binary_mask)
        if(save_path!=''):
            self.save_image(trans_mask, save_path)
        return trans_mask
    
    def __createMask(self, mask_ds:np.ndarray=None) -> tuple:
        """
        Generates a mask for the given image and index, applying various random augmentations if enabled.
        Parameters:
            img (np.ndarray): The input image to be masked.
        Returns:
            tuple:
                mask (np.ndarray): The mask matrix applied to the image.
        Behavior:
            - If a mask dataset (`mask_ds`) is provided, retrieves the mask 
            - If no mask dataset is provided, creates a white mask of the same dimensions as the image.
            - If `want_result.rand_m` is enabled, draws random lines, circles, and ellipses on the mask to create random patterns.
        Raises:
            Exception: If the mask dimensions are less than 64x64 when `want_result.rand_m` is enabled.
        """

        ## Prepare masking matrix
        dim = mask_ds.shape[:2]
        # dim = (self.image_config['model_input'][0],self.image_config['model_input'][1])
        if(mask_ds is not None):
            mask = mask_ds.copy()
        elif(mask_ds is None):
            mask = np.full((dim[0],dim[1],3), 1.0, np.float32) ## White background
        # Set size scale
        if(self.want_result[self.current_mode]['rand_m']):
            size = int((dim[0] + dim[1]) * 0.03)
            if dim[0] < 64 or dim[1] < 64:
                raise Exception("Width and Height of mask must be at least 64!")
            # Draw random lines
            if np.random.choice([True, False]):
                for _ in range(np.random.randint(1, 10)):
                    # Get random x locations to start line
                    x1, x2 = np.random.randint(1, dim[0]), np.random.randint(1, dim[1])
                    # Get random y locations to start line
                    y1, y2 = np.random.randint(1, dim[0]), np.random.randint(1, dim[1])
                    # Get random thickness of the line drawn
                    thickness = np.random.randint(1, 3)
                    # Draw black line on the white mask
                    cv2.line(mask,(x1,y1),(x2,y2),(255,255,255),thickness)
            # Draw random circles
            if np.random.choice([True, False]):
                for _ in range(np.random.randint(1, 20)):
                    x1, y1 = np.random.randint(1, dim[0]), np.random.randint(1, dim[1])
                    radius = np.random.randint(3, size)
                    cv2.circle(mask,(x1,y1),radius,(255,255,255), -1)
            # Draw random ellipses
            if np.random.choice([True, False]):
                for _ in range(np.random.randint(1, 20)):
                    x1, y1 = np.random.randint(1, dim[0]), np.random.randint(1, dim[1])
                    s1, s2 = np.random.randint(1, dim[0]), np.random.randint(1, dim[1])
                    a1, a2, a3 = np.random.randint(3, 180), np.random.randint(3, 180), np.random.randint(3, 180)
                    thickness = np.random.randint(3, size)
                    cv2.ellipse(mask, (x1,y1), (s1,s2), a1, a2, a3,(255,255,255), thickness)

        ## Mask the image
        # masked_image = mask * img + (1 - mask) * 1.0

        return mask

    def flip_image(self,image:np.ndarray, flip_path:str, flip_type:str='horizontal') -> tuple:
        """
        flip image and mask to horizontal or vertical
        """
        # Flip the image
        if flip_type == 'horizontal':
            flipped_image = cv2.flip(np.array(image), 1)
        elif flip_type == 'vertical':
            flipped_image = cv2.flip(np.array(image), 0)
        else:
            raise ValueError("Invalid flip_type. Use 'horizontal' or 'vertical'.")
        
        # Save the flipped image
        # self.save_image(flipped_image,self.output_path + '_flip.jpg')
        # self.save_image(binary_change())
        self.save_image(flipped_image, flip_path + '_flip.jpg')
        # self.binary_trans_mask(flipped_mask,self.output_path+ '_flip_mask.png')
        return flipped_image

    def crop_top_and_bottom_horizontal(self, image: np.ndarray, name: str) -> None:
        """
        Crop image into 3 sections (Top, Center, Bottom) and save them along with their masks.
        """
        mask_path = self.mask_path.copy()
        crop_size = self.image_config['model_input'][0]
        height, width = image.shape[:2]
        
        # y_start = (height - crop_size) // 2
        # y_end = y_start + crop_size

        y_crops = {
            "top": (0, crop_size),
            "center_y": ((height - crop_size) // 2, (height + crop_size) // 2),
            "bottom": (height - crop_size, height)
        }
        x_crops = {
            "left": (0, crop_size),
            "center_x": ((width - crop_size) // 2, (width + crop_size) // 2),
            "right": (width - crop_size, width)
        }
        result_img = {
            "top": None,
            "center_y": None,
            "bottom": None
        }

        for y_crop_name, (y_start, y_end) in y_crops.items():
            y_want_crop = (
                y_crop_name == "top" and self.want_result[self.current_mode]['height_crop'].get('crop_T')
                or y_crop_name == "center_y" and self.want_result[self.current_mode]['height_crop'].get('crop_Cy')
                or y_crop_name == "bottom" and self.want_result[self.current_mode]['height_crop'].get('crop_B')
            )
            if y_want_crop:
                x_result_img = {
                    "left": (None, None),
                    "center_x": (None, None),
                    "right": (None, None)
                }
                for x_crop_name, (x_start, x_end) in x_crops.items():
                    x_want_crop = (
                        x_crop_name == "left" and self.want_result[self.current_mode]['width_crop'].get('crop_L')
                        or x_crop_name == "center_x" and self.want_result[self.current_mode]['width_crop'].get('crop_Cx')
                        or x_crop_name == "right" and self.want_result[self.current_mode]['width_crop'].get('crop_R')
                    )
                    if x_want_crop:
                        crop_img = image[y_start:y_end, x_start:x_end]
                        self.save_image(crop_img, f"{self.output_path}_{y_crop_name}_{x_crop_name}_crop{name}.jpg")
                        mask = crop_center(
                            mask_path[np.random.randint(0, self.len_mask)].copy(),
                            self.image_config['model_input']
                        )
                        mask = self.__createMask(mask)
                        mask = self.binary_trans_mask(mask, f"{self.output_path}_{y_crop_name}_{x_crop_name}_crop{name}_mask.png")
                        x_result_img[x_crop_name] = (crop_img, mask)
                result_img[y_crop_name] = x_result_img
        return result_img

    def crop_top_and_bottom_vertical(self, image: np.ndarray, name: str) -> None:
        """
        Crop image into 3 sections (Top, Center, Bottom) and save them along with their masks.
        """
        mask_path = self.mask_path.copy()
        crop_size = self.image_config['model_input'][0]
        height, width = image.shape[:2]
        
        x_start = (width - crop_size) // 2
        x_end = x_start + crop_size
        crop_img = None
        mask = None

        crops = {
            "top_crop": (0, crop_size),
            "center_crop": ((height - crop_size) // 2, (height + crop_size) // 2),
            "bottom_crop": (height - crop_size, height)
        }
        result_img = {
            "top_crop": (None, None),
            "center_crop": (None, None),
            "bottom_crop": (None, None)
        }

        for crop_name, (y_start, y_end) in crops.items():
            crop_img = None
            mask = None
            want_crop = (
                crop_name == "top_crop" and self.want_result[self.current_mode]['height_crop'].get('crop_T')
                or crop_name == "center_crop" and self.want_result[self.current_mode]['height_crop'].get('crop_Cy')
                or crop_name == "bottom_crop" and self.want_result[self.current_mode]['height_crop'].get('crop_B')
            )
            if want_crop:
                crop_img = image[y_start:y_end, x_start:x_end]
                self.save_image(crop_img, f"{self.output_path}_{crop_name}{name}.jpg")
                mask = crop_center(
                        mask_path[np.random.randint(0, self.len_mask)].copy(),
                        self.image_config['model_input']
                    )
                mask = self.__createMask(mask)
                mask = self.binary_trans_mask(mask, f"{self.output_path}_{crop_name}{name}_mask.png")
                result_img[crop_name] = (crop_img, mask)
                crop_img = None
                mask = None
        crop_img = None
        mask = None
        return result_img

    def equi_to_per(self, car_pano_img:np.ndarray) -> np.ndarray:
        """
        Converts an equirectangular panoramic image to a perspective view.
        Args:
            car_pano_img (np.ndarray): Input equirectangular panoramic image as a NumPy array.
        Returns:
            np.ndarray: Perspective view image generated from the input panorama.
        Notes:
            - The method uses the Perspective class to perform the transformation.
            - The perspective parameters (FOV, theta, phi, height, width) are set internally.
        """
        equ = Perspective(car_pano_img)    # Load equirectangular image
        per_img_1000 = equ.GetPerspective(167, self.image_config['coor'], -90, self.image_config['img_top_h'], self.image_config['img_top_w'])  # Specify parameters(FOV, theta, phi, height, width)
        return per_img_1000
    
    def equi_to_cube(self, car_pano_img:np.ndarray) -> tuple:
        out = e2c(car_pano_img, face_w=1000, mode='bilinear', cube_format='dict') # [front, right, back, left, top, bottom]
        keys_to_check = ['F', 'R', 'B', 'L', 'U', 'D']
        new_name = ['cube-front', 'cube-right', 'cube-back', 'cube-left', 'cube-top', 'cube-bottom']
        # Get keys as a list
        keys = list(out.keys())
        # Define range that overlay function want
        # start = keys.index(keys_to_check[0])
        # end = keys.index(keys_to_check[-1])
        # if edit mask on mask_page set new mask
        selected = {new_name[i]: out[k] for i, k in enumerate(keys)}
        return selected
        # save image
        # for i in range(len(cube_face)):
        #     out_image_im = Image.fromarray(((out[cube_face[i]])).astype(np.uint8))
        #     out_image_im.save(save_path + name_img[0] +'_'+ cube_face[i] +'.'+ name_img[1])

    
    def pipeline_aug(self) -> None:
        mask_path = self.mask_path.copy()
        img_name = []
        out_image = [] # stored perspective (top view image)
        car_pano_img = preprocess_img(self.img_path,img_size=(self.image_config['img_pano_w'],self.image_config['img_pano_h'])) # read image file
        if(self.image_mode["per"]):
            self.current_mode = "per"
            per_image = self.equi_to_per(car_pano_img) # get perspective image
            mask_image = cv2.resize(mask_path[np.random.randint(0, self.len_mask)].copy(),self.image_config['model_input'])
            mask_image = self.__createMask(mask_image)
            mask_image = self.binary_trans_mask(mask_image)
            self.save_image(mask_image,self.output_path+ '_per_mask.png')
            self.save_image(per_image, self.output_path+'_per.jpg') # save perspective image
            out_image.append(per_image) 
            img_name.append('')
            if(self.want_result["per"]['flip']): # if want to flip
                per_flip = self.flip_image(per_image, self.output_path+'_per') # flip image and save
                mask_flip = cv2.resize(mask_path[np.random.randint(0, self.len_mask)].copy(),self.image_config['model_input'])
                mask_flip = self.__createMask(mask_flip)
                mask_flip = self.binary_trans_mask(mask_flip)
                self.save_image(mask_flip, self.output_path+'_per_flip_mask.png')
                out_image.append(per_flip)
                img_name.append('_flip')
            for img, name in zip(out_image, img_name):
                self.crop_top_and_bottom_vertical(img, name) # get 3 crop (top, center, bottom) from perspective image
        if(self.image_mode["cube"]):
            self.current_mode = "cube"
            cube_image = self.equi_to_cube(car_pano_img)
            for key, img in cube_image.items():
                if(self.want_result["cube"]["side"].get(key)):
                    img_name = []
                    out_image = [] # stored perspective (top view image)
                    mask_image = cv2.resize(mask_path[np.random.randint(0, self.len_mask)].copy(),self.image_config['model_input'])
                    mask_image = self.__createMask(mask_image)
                    mask_image = self.binary_trans_mask(mask_image)
                    self.save_image(mask_image,self.output_path+ '_'+key+'_mask.png')
                    self.save_image(img, self.output_path+'_'+key+'.jpg') # save perspective image
                    out_image.append(img) 
                    img_name.append('')
                    if(self.want_result["cube"]['flip']): # if want to flip
                        cube_flip = self.flip_image(img, self.output_path+'_'+key) # flip image and save
                        mask_image = cv2.resize(mask_path[np.random.randint(0, self.len_mask)].copy(),self.image_config['model_input'])
                        mask_image = self.__createMask(mask_image)
                        mask_image = self.binary_trans_mask(mask_image)
                        self.save_image(mask_image,self.output_path+ '_'+key+'_mask.png')
                        out_image.append(cube_flip)
                        img_name.append('_flip')
                    for name, y_img in zip(img_name, out_image):
                        tmp_result = self.crop_top_and_bottom_horizontal(y_img, '_'+key+name) # get 3 crop (top, center, bottom) in y-axis


# Example usage
# want_ds = ['C1','C2','Br']
# path_ds = ["./car_ds/pic/C1_output_new/","./car_ds/pic/C2_output_new/","./car_ds/pic/image_test2/"]
# format_name = ['img_','img_','']
# format_file = ['_O.jpg','_O.jpg','_afterfill.jpg']
# format_name_num = [6, 6, 1]
# save_path = "./car_ds/pic/aug_ds/"
# mask_path = ['./car_ds/pic/mask/move_8k_focus_simple2_1000_crop.jpg',
#              './car_ds/pic/mask/move_binary_mask_crop.jpg',
#              './car_ds/pic/mask/binary_mask_test_crop.jpg',
#              './car_ds/pic/mask/8k_focus_simple2_crop.jpg']
def main():
    aug_path = os.path.join(pathlib.Path(__file__).parent.absolute(),'aug.yml')
    with open(aug_path, 'r') as f:
        FLAGS = yaml.safe_load(f)
    path_ds = os.path.join(pathlib.Path(__file__).parent.absolute(), "car_ds/pic/"+_argparse().dir+'/')
    save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "car_ds/pic/"+_argparse().save+'/')
    create_fol(save_path)
    config_name = _argparse().conf
    img_mode = FLAGS['img_mode']
    mask_path = glob.glob(os.path.join(pathlib.Path(__file__).parent.absolute(), "car_ds/pic/mask/")+'*'+'.jpg')
    want_result = {
        "per": FLAGS['per'],
        "cube": FLAGS['cube']
    }
    setting_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "system_settings.json")
    with open(setting_path , 'r') as f:
        settings = json.load(f)
    image_config_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "config_gan_ui.json")
    with open(image_config_path ,'r') as f:
        image_config = json.load(f)

    format_file = settings['global']['output']['suffix_format']
    use_image_config = image_config[config_name]
    # for count, main_path in enumerate(path_ds):
    # target = []
    # main_path = "./car_ds/pic/C1_output_new/"  # Replace with your input image path
    dir_list_input = glob.glob(path_ds+'*'+format_file+'.jpg')
    for i in range(len(dir_list_input)):
        name_file = ((dir_list_input[i].replace('\\','/')).split('/'))[-1]
        # if(format_name_num[count] == 6):
        #     num_file = ((name_file.split('_'))[1])
        # elif(format_name_num[count] == 1):
        #     num_file = ((name_file.split('_'))[0])
        # target.append(int(num_file))
        # range_shift = [1,-1]
        # print('name_file: ',name_file,' num_file: ',num_file)
        # input_path = main_path + format_name[count] + str(num_file).rjust(format_name_num[count] , '0') + format_file[count]
        input_path = dir_list_input[i]
        output_path = save_path + name_file.split('.')[0]  # Replace with your desired output path
        try:
            data_aug = process_aug(input_path, output_path, use_image_config, mask_path,img_mode,want_result)
            data_aug.pipeline_aug()
            # equi_to_per(image_config)
            # crop_top_and_bottom(input_path, output_path)
            # break
        except FileNotFoundError:
            print(f"File not found: {input_path}")
            continue

if __name__ == "__main__":
    main()

# binary_change(mask_path[np.random.randint(0, 4)],'./car_ds/pic/aug_ds/test'+ '_top_crop_mask.png')

        # image_name =  "img_"+str(source_path).rjust(6, '0')+".jpg"

# shift the crop area vertically
# for j in range_shift:
#     for i in range(0,2):
#         vertical_offset = 244 * i * j  # Move the crop area 50 pixels down
#         output_path = "./car_ds/pic/aug_ds/n_c1_000000_O_m"+str(vertical_offset)+".jpg"  # Replace with your desired output path
#         crop_and_save_image(input_path, output_path, vertical_offset)