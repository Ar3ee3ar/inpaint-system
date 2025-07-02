import numpy as np
import cv2
from PIL import Image
import time
from typing import Union
import utm
# import matplotlib.pyplot as plt
# check corrupted image
# from skimage import io

from . import find_file_path
from .utils import binary_change, up_downscale, gen_tranparent_mask
from .process_pano import Equirectangular,Perspective,inpaint_pano
from .process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from .dataset import Dataset, createAugment
from .py360convert.py360convert import c2e, e2c

# -------------------- describe process for inpainting -----------
# inpainting have 2 approach :
# 1. pad-image: 
#   address limit of inpainting algorithm that not suit for large image 
#   (this work case is 8k panoramic images) by select only limited area. 
#   this method will make sure final output is not lose resolution of image
#   - panoramic projection into perspective / cube map
#   - target selection: 
#       crop image to focus only target object (called crop center). 
#       the crop image size will match model input size.
#       in case of cube map it will focus only center of image
#   - inpainting 
#       (in this case mask hole will have much more larger size than other approach)
#   - overlay process: 
#       padding image with black pixel (unwanted inpainted area) 
#       to make inpainted image match size of perspective/ map then use 
#       cv2 seamless that will detect center of white pixel (wanted inpainted area) 
#       by using cv2.boundingRect then use it to specify center of image that will 
#       overlay inpainted area to original image 
#   - perspective / cube map projection into perspective image
#   Notes:
#       - may not appropriate for work to detect other object in cube map
#       - model might not inpainted area to fullest due to large hole problem
# 2. resize:
#   address limit in previous method (pad-image) that not suit for detect object
#   that in other position (not in center of image) and large hole problem
#   - panoramic projection into perspective / cube map
#   - resize original image: 
#       decreased image size to match model input size same as cube map case.
#       by using cv2.resize
#   - inpainting 
#       (in this case mask hole will have smaller size than other approach)
#   - overlay process:
#       increased image size to match model input size: have 2 approach
#       - cv2 resize: use neighbor/ bicubic to increased image size
#       - RealESRGAN: Pre-trained GAN for increase image by not losing resolution
#           (Notes: this method make image not lose resolution but not increased 
#               resolution)
#       to make inpainted image match size of perspective/ map then use 
#       cv2 seamless that will detect center of white pixel (wanted inpainted area) 
#       by using cv2.boundingRect then use it to specify center of image that will 
#       overlay inpainted area to original image 
#   - perspective / cube map projection into perspective image
#   Notes:
#       Need to find algorithm for increased size and not lose resolution


# ------------------- bird view overlay function -------------------------

class OverlayGAN():
    """
    OverlayGAN is a class designed for overlaying and inpainting images using GAN-based models, particularly in the context of panoramic and perspective image transformations. It supports mask management, seamless cloning, and conversion between equirectangular and perspective projections.
    Attributes:
        config_data (dict): Configuration data containing image and mask parameters.
        coor: Coordinates for image projection.
        mask: Mask configuration.
        img_top_h (int): Height of the top-view image.
        img_top_w (int): Width of the top-view image.
        img_pano_h (int): Height of the panorama image.
        img_pano_w (int): Width of the panorama image.
        move_img_up: Parameter for moving the image up.
        move_img_down: Parameter for moving the image down.
        x_new_coor: New x-coordinate for projection.
        y_new_coor: New y-coordinate for projection.
        relate_dist: Related distance parameter.
        model_input (tuple): Input size for the model.
        model_type (str): Type of the model (e.g., "p2p", "df").
        model_weight: Model weights.
        type_process_scale (str): Scaling method for processing images.
        mask_img: Processed mask image.
        in_mask: Input mask for the model.
        mask_img_seamless: Mask image for seamless cloning.
        mask_pano_img: Mask image in panorama format.
        df_mask: Transparent mask for deep fill models.
        have_mask (bool): Status indicating if a mask is present.
        generator: The GAN generator model.
    Methods:
        __init__(config_data, config_name, type_process_scale='resize'):
            Initializes the OverlayGAN object with configuration data.
        set_model_generator(generator):
            Sets the GAN generator model.
        get_haveMask_status():
            Returns the status indicating if a mask is present.
        set_segMask(seg_mask):
            Sets the segmentation mask and processes it for model input and panorama projection.
        set_useMask():
            Loads and processes the mask from the configuration for use in the model and panorama.
        seamless(source, target, mask):
            Performs seamless cloning of the source image onto the target using the provided mask.
        equi_to_per(img_path, name=''):
            Converts an equirectangular image to a perspective image and optionally saves it.
        save_pano(inpaint_fill, save_path, count):
            Saves the inpainted panorama image to the specified path.
        per_to_equi(car_path, per_img):
            Projects a perspective image back to equirectangular format and applies inpainting.
        crude_inpaint(source, target):
            Performs a simple inpainting by blending the source and target images using the mask.
        overlay(main_path, target_coor):
            Main method to perform overlay and inpainting for a given image and coordinate.
    Notes:
        - This class relies on external functions and classes such as preprocess_img, up_downscale, Equirectangular, Perspective, createAugment, gen_tranparent_mask, and inpaint_pano.
        - The class is designed for use in image inpainting pipelines, particularly for panoramic imagery.
    """

    def __init__(self, config_data: dict, config_name: str, type_process_scale: str = 'resize'):
        """
        Initializes the OverlayGAN object with configuration data.
        """
        print('Start read '+config_name+' config...')
        self.config_data = config_data
        self.coor = config_data["coor"]
        self.mask = config_data["mask"]
        self.img_top_h = config_data["img_top_h"]
        self.img_top_w = config_data["img_top_w"]
        self.img_pano_h =  config_data["img_pano_h"]
        self.img_pano_w = config_data["img_pano_w"]
        self.move_img_up = config_data["move_img_up"]
        self.move_img_down = config_data["move_img_down"]
        self.x_new_coor = config_data["x_new_coor"]
        self.y_new_coor = config_data["y_new_coor"]
        self.relate_dist = config_data["relate_dist"]
        self.model_input = config_data["model_input"]
        self.model_type = config_data["model_type"]
        self.model_weight = config_data["model_weight"]
        self.type_process_scale = type_process_scale
        # ------- mask value -----------------------
        self.mask_img = None
        self.in_mask = None
        self.mask_img_seamless = None
        self.mask_pano_img = None
        self.df_mask = None
        self.have_mask = False
        # -------- Generator value -----------------
        self.generator = None

        print('Complete reading config '+config_name)

    def set_model_generator(self, generator: object) -> None:
        """
        Sets the model generator object to be used.
        Args:
            generator (object): The generator object to assign to the model.
        Returns:
            None
        """
        self.generator = generator

    def get_haveMask_status(self) -> bool:
        """
        Returns the current status indicating whether a mask is present.
        Returns:
            bool: True if a mask is present, False otherwise.
        """
        return self.have_mask
    
    def set_useMask(self) -> None:
        """
        Prepares and processes the mask image for use in the model and overlay operations.
        This method performs the following steps:
        1. Loads and resizes the mask image if a mask path is provided; otherwise, creates a blank mask.
        2. Binarizes the mask image and sets the 'have_mask' flag accordingly.
        3. Stores the full-sized mask image and prepares a cropped or downscaled version to match the model input size.
        4. Preprocesses the mask for model input, including bitwise inversion for U-Net compatibility.
        5. Projects the mask onto an equirectangular panorama for overlay purposes.
        6. Generates a transparent mask for further processing and converts it to grayscale and normalized float32 format.
        Attributes set:
            - self.have_mask: Boolean indicating if a mask is present.
            - self.mask_img: The full-sized mask image.
            - self.in_mask: The processed mask for model input.
            - self.mask_img_seamless: The seamless mask image.
            - self.mask_pano_img: The mask projected onto a panorama.
            - self.df_mask: The transparent, normalized mask for downstream tasks.
        """
        # ---------- load image -------------------------
        if(self.mask["top_view"] != ""):
            mask_img = cv2.resize(cv2.imread(find_file_path(self.mask["top_view"])),(self.img_top_h,self.img_top_w)) # for model
            mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8) # binarize image
            self.have_mask = True
        else:
            mask_img = np.full((self.img_top_h,self.img_top_w,3), 0.0, np.float32) # blank mask (all black)
            self.have_mask = False
        self.mask_img = mask_img/255.0 # full mask (match original input size) - unwanted black px (0) - wanted white px (1) [0,1]
        # ------------- preprocess for model input ---------------------
        # crop (downscale) to match model input size | resize (downscale) to match model input size -----
        mask_img_crop = up_downscale(mask_img,(self.model_input[0],self.model_input[1]),'downscale',self.type_process_scale)
        mask_img_crop = preprocess_img(mask_img_crop,img_size=(self.model_input[0],self.model_input[1]))
        mask_bw = cv2.bitwise_not(mask_img_crop) # for only u-net (p2p) - unwanted white px (255) - wanted white px (0)
        self.in_mask = mask_bw.astype('float32') # in range of [0,255]
        # --------------- preprocess mask for overlay panoramic image -----------
        mask_img_full = mask_img # get original input mask
        mask_img_full = np.where(mask_img_full > 127, 255, 0).astype(np.uint8) # binarized
        # ------------------ seamless mask -----------------------------
        self.mask_img_seamless = mask_img # get original input size
        # project to equirectangular (get mask image) -------------------
        equ = Equirectangular(mask_img_full, 167, self.coor, -90,'img')    
        mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        self.mask_pano_img = mask_pano_img.astype(np.uint8)
        # ---------------- input mask for deepfill (df) ----------------
        # Generate a transparent mask for deep fill models using the binarized, cropped mask
        self.df_mask = gen_tranparent_mask(binary_change(mask_img_crop))
        # If the generated mask has an alpha channel (4 channels), convert it to grayscale
        if self.df_mask.shape[-1] == 4:  # Check if the mask has an alpha channel
            self.df_mask = cv2.cvtColor(self.df_mask, cv2.COLOR_RGBA2GRAY)
        # Add a channel dimension to the mask (shape: H x W x 1)
        self.df_mask = np.expand_dims(self.df_mask, 2)
        # Add a batch dimension to the mask (shape: 1 x H x W x 1), convert to float32
        self.df_mask = np.expand_dims(self.df_mask, 0).astype(np.float32)
        # Normalize the mask values to the range [0, 1]
        self.df_mask = (self.df_mask/255.0).astype(np.float32)

    def set_segMask(self,seg_mask:np.ndarray) -> None:
        """
        Sets and processes a segmentation mask for model input and panorama projection.
            seg_mask (np.ndarray): A numpy array representing the segmentation mask, 
            expected in the range [0, 255].
        Behavior:
            - If a mask is provided, it is processed and stored; otherwise, a blank mask is created.
            - Converts the mask to a binary format and generates both inverted (for pix2pix(p2p)) 
                and original versions.
            - Resizes the mask to match the original input size and prepares it for seamless blending.
            - Projects the mask onto a panorama using equirectangular projection.
            - Generates a transparent mask suitable for model input (for deepfill(df)), 
                converting it to grayscale and normalizing.
        Notes:
            Use this method when a custom mask (e.g., from user input) is needed 
            instead of the default configuration.
        """
        # ---------- load image -------------------------
        if(seg_mask is not None):
            mask = np.array(seg_mask) # grayscale to ensure image is binary
            self.have_mask = True
        else:
            mask = np.full((self.model_input[0],self.model_input[1],3), 0.0, np.float32)
            self.have_mask = False
        # ------------- preprocess for model input ---------------------
        mask_ori = binary_change(mask[:]) # binarized
        mask_bw = cv2.bitwise_not(mask_ori[:]) # reversed original for p2p - unwanted white px (255) - wanted black px (0)
        self.in_mask = mask_bw # for p2p
        # padding (upscale) to match original input size | resize (upscale) to match original input size -----
        pad_mask = up_downscale((mask_ori/255.0).astype(np.float32),(self.img_top_w,self.img_top_h),'upscale',self.type_process_scale)
        self.mask_img = pad_mask # original mask size [0,1]
        self.mask_img_seamless = (pad_mask[:]*255.0).astype(np.uint8) # for seamless in range of[0,255]
        # ----------- preprocess mask for overlay panoramic image ---------------
        equ = Equirectangular((pad_mask[:]*255.0).astype(np.float32), 167, self.coor, -90,'img') # in range of [0,255]   
        mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        self.mask_pano_img = mask_pano_img.astype(np.uint8)
        # ---------------- input mask for deepfill (df) ----------------
        # Generate a transparent mask for deep fill models using the binarized, cropped mask
        self.df_mask = gen_tranparent_mask(mask_ori)
        # If the generated mask has an alpha channel (4 channels), convert it to grayscale
        if self.df_mask.shape[-1] == 4:  # Check if the mask has an alpha channel
            self.df_mask = cv2.cvtColor(self.df_mask, cv2.COLOR_RGBA2GRAY)
        # Add a channel dimension to the mask (shape: H x W x 1)
        self.df_mask = np.expand_dims(self.df_mask, 2)
        # Add a batch dimension to the mask (shape: 1 x H x W x 1), convert to float32
        self.df_mask = np.expand_dims(self.df_mask, 0).astype(np.float32)
        # Normalize the mask values to the range [0, 1]
        self.df_mask = (self.df_mask/255.0).astype(np.float32)

    def seamless(self,source:np.ndarray, target:np.ndarray, mask:np.ndarray) -> np.ndarray:
        """
        Performs seamless cloning of a source image onto a target image using a specified mask.
        This method computes the bounding rectangle of the mask to determine the center point for seamless cloning.
        It then blends the source image into the target image at the computed center using OpenCV's seamlessClone function.
        Args:
            source (np.ndarray): The source image to be cloned onto the target.
            target (np.ndarray): The target image where the source will be blended.
            mask (np.ndarray): The mask defining the region of the source image to be cloned.
        Returns:
            np.ndarray: The resulting image after seamless cloning.
        """

        x, y, w, h = cv2.boundingRect(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

        # Compute center of the bounding box
        center = (x + w // 2, y + h // 2)
        # Get seamless center from config data setting
        # (h, w) = (self.seamless_center_h, self.seamless_center_w)
        # center = (h,w)
        # center = (1002,901)
        
        im_clone = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)

        return im_clone

    # def save_per(self,img_path:np.ndarray, save_path:str, count:str) -> None:
    #     """
    #     Processes an input image, generates a perspective-transformed top view, and saves the result as a JPEG file.
    #     Args:
    #         img_path (np.ndarray): The input image array to be processed.
    #         save_path (str): The directory path where the output image will be saved.
    #         count (str): A string identifier to append to the output filename.
    #     Returns:
    #         None
    #     Notes:
    #         for debugging perspective result
    #     """
    #     car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
    #     equ = Perspective(car_pano_img) 
    #     per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)
    #     input_image_im = Image.fromarray((per_img_1000).astype(np.uint8))
    #     input_image_im.save(save_path+count+"_topview_source.jpg")
    

    def equi_to_per(self,car_pano_img:np.ndarray) -> np.ndarray:
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
        # car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
        equ = Perspective(car_pano_img)    # Load equirectangular image
        per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)  # Specify parameters(FOV, theta, phi, height, width)
        return per_img_1000
    
    def save_pano(self,inpaint_fill:np.ndarray, save_path:str, count:str) -> None:
        """
        Saves an inpainted panorama image to the specified path.
        Args:
            inpaint_fill (np.ndarray): The inpainted image as a NumPy array with values in [0, 1].
            save_path (str): The directory path where the image will be saved.
            count (str): A string or number used to generate a zero-padded filename.
        Returns:
            None
        """
        impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
        impainted_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg")

    def per_to_equi(self,car_pano_img:np.ndarray, per_img:np.ndarray) -> np.ndarray:
        """
        Projects a perspective image onto an equirectangular panorama and performs inpainting.
        Args:
            car_pano_img (np.ndarray): The original panoramic image of the car.
            per_img (np.ndarray): The perspective image to be projected onto the panorama.
        Returns:
            np.ndarray: The inpainted panoramic image after projecting the perspective image and applying inpainting.
        """
        equ = Equirectangular(per_img,167, self.coor, -90)    
        inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
        inpaint_fill = inpaint_pano(inpaint_pano_img,self.mask_pano_img,car_pano_img)
        return inpaint_fill

    def crude_inpaint(self,source:np.ndarray, target:np.ndarray) -> np.ndarray:
        """
        Performs a crude inpainting operation by blending the source and target images using a binary mask.
        The method uses the instance's mask image (`self.mask_img`) to determine which pixels to take from the source image
        and which from the target image. Pixels where the mask is non-zero are taken from the source, and the rest from the target.
        Args:
            source (np.ndarray): The source image as a NumPy array, expected to be in the same shape as the target.
            target (np.ndarray): The target image as a NumPy array.
        Returns:
            np.ndarray: The resulting image after blending source and target according to the mask.
        """
        mask = self.mask_img
        reverse_mask = cv2.bitwise_not(mask)
        source = (source/255.0).astype(np.float32)
        target = (target/255.0).astype(np.float32)
        reverse_mask[reverse_mask != 0] = 1
        result = reverse_mask * target + (1 - reverse_mask) * source
        return result
    
    def overlay(self,main_path: Union[str,np.ndarray],target_coor:int) -> np.ndarray:
        """
        Overlays an inpainted or processed region onto a panoramic image at a specified coordinate.
        Depending on the model type ('p2p' for pix2pix or 'df' for deepfill), this method loads or receives an image,
        preprocesses it, applies the selected inpainting/generation model, and blends the result back into the original
        panoramic image using seamless cloning.
        Args:
            main_path (Union[str, np.ndarray]): Path to the main image directory or a numpy array representing the image.
            target_coor (int): The target coordinate index used to select or process the image.
        Returns:
            np.ndarray: The final panoramic image with the inpainted or processed region overlaid.
        Raises:
            FileNotFoundError: If the image file cannot be found when a path is provided.
            ValueError: If the model type is not supported or required attributes are missing.
        Notes:
            - The method expects several instance attributes to be defined, such as image dimensions, model type,
              generator, masks, and processing parameters.
            - The function supports both pix2pix and deepfill inpainting models.
            - Images are preprocessed, normalized, and resized as required by the model and then post-processed
              to match the original panorama's dimensions.
        """
        # load image with preprocess -------------------------------
        if isinstance(main_path, str):
            img_path = main_path + "img_"+str(target_coor).rjust(6, '0')+".jpg"
            # load image with preprocess
            car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
        else:
            car_pano_img = (np.array(main_path*255)).astype(np.uint8)
        # project to perspective (get car image)
        per_img_1000 = self.equi_to_per(car_pano_img)
        # car image
        # crop image (downscale) to match model input size | resize (downscale) to match model input size -----
        per_img = up_downscale(per_img_1000,(self.model_input[0], self.model_input[1]),'downscale',self.type_process_scale)
        car_img_top = preprocess_img(per_img,img_size=(self.model_input[0],self.model_input[1])) # reprocess image for guarantee range of data
        car_img_top = car_img_top.astype('float32')
        # pix2pix ------------------------------------
        if(self.model_type == "p2p"):
            testgen = createAugment(np.array([car_img_top]), np.array([car_img_top]),np.array([self.in_mask]),batch_size=1,dim=(self.model_input[0],self.model_input[1]), shuffle=False, random_mask=False)
            [masked_image, mask], car_img = testgen[0]
            car_img = (car_img - 0.5)/0.5 # normalize [-1,1]
            masked_image = (masked_image - 0.5)/0.5 # normalize [-1,1]
            inputs = masked_image # pix2pix
            predict_image = self.generator(inputs, training=True)
            predict_image_norm = ((predict_image[0]* 0.5) + 0.5).numpy()# de-normalize [0,1]
            masked_image = (masked_image* 0.5) + 0.5 # de-normalize [0,1]
            car_img = (car_img* 0.5) + 0.5 # de-normalize [0,1]
            # ------------------------------------------------
        # deepfill (df) ----------------------------------------
        elif(self.model_type == "df"):
            image = np.expand_dims(car_img_top, 0).astype(np.float32)
            image = image / 255.0
            masked_image = image*(1. - self.df_mask)
            x1, x2, offset_flow = self.generator(masked_image, self.df_mask, training=False)
            batch_predicted = x2
            result = batch_predicted  * self.df_mask + masked_image * (1. - self.df_mask)
            predict_image_norm = np.array(batch_predicted[0])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        per_img_1000 = (per_img_1000/255.0).astype(np.float32)
        predict_image_norm = preprocess_img(predict_image_norm,img_size=(self.model_input[0],self.model_input[1])) # reprocess image for guarantee range of data
        # padcrop image (upscale) to match original input size | resize (upscale) to match original input size -----
        predict_image_norm = up_downscale(predict_image_norm,(self.img_top_w,self.img_top_h),'upscale',self.type_process_scale) 
        # predict_image_norm = ((1 - mask_img_pano) * per_img_1000) + (mask_img_pano * predict_image_norm) # can use for crude inpaint without calling crude_inpaint function | white wanted px (0)
        predict_image_norm = self.seamless((predict_image_norm*255.0).astype(np.uint8),(per_img_1000*255.0).astype(np.uint8),self.mask_img_seamless)
        final_pano_output = self.per_to_equi(car_pano_img,predict_image_norm)
        return final_pano_output

# --------------- inherited for cube overlay -------------------------
class OverlayCube(OverlayGAN):
    """
    OverlayCube is a subclass of OverlayGAN designed for handling cube map overlays and inpainting tasks, particularly for 360-degree images or panoramas. It manages masks and images for each face of a cube (front, right, back, left, top, bottom), processes them for use with GAN-based inpainting models, and projects between cube and equirectangular (panorama) representations.
    Attributes:
        generator: The GAN model or generator used for inpainting.
        in_mask (dict): Stores binary masks for each cube face, resized for model input.
        mask_img (dict): Stores mask images for each cube face, resized to the original face size.
        mask_img_seamless (dict): Stores seamless mask images for each cube face.
        df_mask (dict): Stores transparent masks for each cube face, processed for model input.
        have_mask (dict): Flags indicating if a mask is present for each cube face.
        type_process_scale (str): Method for resizing images and masks.
        blank_side (np.ndarray): Placeholder image for faces without a mask.
    Methods:
        set_useMask():
            Loads and processes mask images for each cube face, preparing them for model input and seamless overlay. Generates binary and transparent masks, and projects the masks to an equirectangular panorama.
        set_segMask(seg_mask):
            Sets segmentation masks for each cube face from provided data, processes them for model input, and updates the panorama mask.
        per_to_equi(car_path, per_img):
            Projects a perspective or cube face image back to an equirectangular panorama, applying inpainting using the current panorama mask.
        overlay(pic_path, target_coor):
            Main method to perform overlay/inpainting on a panorama. Projects the panorama to cube faces, applies the inpainting model to each masked face, reconstructs the panorama, and blends the result seamlessly.
    Note:
        This class relies on several external utility functions (e.g., preprocess_img, up_downscale, binary_change, gen_tranparent_mask, c2e, e2c, inpaint_pano, createAugment) and expects certain attributes (e.g., self.img_top_h, self.img_top_w, self.img_pano_h, self.img_pano_w, self.model_input, self.type_process_scale, self.model_type, self.coor) to be defined in the parent class or during initialization.
    """

    def __init__(self,config_data:dict, config_name:str, type_process_scale: str = 'resize'):
        super().__init__(config_data, config_name,type_process_scale)
        self.generator = None
        self.in_mask = {
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.mask_img = {
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.mask_img_seamless = {
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.df_mask = {
                "cube_front": None,
                "cube_right": None,
                "cube_back": None,
                "cube_left": None,
                "cube_top": None,
                "cube_bottom": None
            }
        self.have_mask = {
                "cube_front": False,
                "cube_right": False,
                "cube_back": False,
                "cube_left": False,
                "cube_top": False,
                "cube_bottom": False
            }
        self.blank_side = np.full((self.img_top_h,self.img_top_w,3), 0.0, np.float32)
        # self.type_process_scale = type_process_scale

    def set_useMask(self) -> None:
        """
        Processes and prepares mask images for each side of a cube map for use in image inpainting models.
        For each side in 'self.mask_img':
            - Loads and resizes the mask image if a mask path is provided, otherwise creates a blank mask.
            - Updates 'self.have_mask' to indicate the presence of a mask for each side.
            - Preprocesses the mask by resizing and normalizing it to match the model input size.
            - Converts the mask to binary and stores it in 'self.in_mask' for U-Net models.
            - Binarizes the full-size mask for overlay and seamless processing.
            - Generates a transparent mask for deep fill models, ensuring it is single-channel and normalized.
            - Stores the processed masks in 'self.df_mask' for each side.
        Finally, projects all processed mask images to an equirectangular panorama and stores the result in `self.mask_pano_img`.
        Attributes modified:
            self.have_mask: Dict indicating if a mask exists for each side.
            self.mask_img: Dict of processed mask images for each side.
            self.in_mask: Dict of binary masks for U-Net models (p2p) wanted black px (0).
            self.mask_img_seamless: Dict of seamless masks for each side.
            self.df_mask: Dict of transparent, normalized masks for deep fill models.
            self.mask_pano_img: Equirectangular panorama of all mask images.
        """

        for count, side_name in enumerate(self.mask_img):
            # load mask ----------------------------
            if(self.mask[side_name] != ""):
                mask_img = cv2.resize(cv2.imread(find_file_path(self.mask[side_name])),(self.img_top_h,self.img_top_w)) # for model
                mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
                self.have_mask[side_name] = True
            else:
                mask_img = np.full((self.img_top_h,self.img_top_w,3), 0.0, np.float32)
                self.have_mask[side_name] = False
            self.mask_img[side_name] = mask_img # read from full top mask img by size of (self.img_top_h,self.img_top_w) - original input image
            # preprocess mask -----------------------------------------------
            # crop (downscale) to match model input size | resize (downscale) to match model input size
            mask_img_crop = up_downscale(mask_img,(self.model_input[0],self.model_input[1]),'downscale',self.type_process_scale) # resize (downscale) to match model input size
            mask_img_crop = preprocess_img(mask_img_crop,img_size=(self.model_input[0],self.model_input[1])) # guarantee value in range of [0,255] with range of model input size
            mask_bw = binary_change(cv2.bitwise_not(mask_img_crop)) # for only u-net (p2p)
            self.in_mask[side_name] = mask_bw.astype('float32') # for only u-net (p2p)
            # for overlay pano ------------------------------------------------------
            mask_img_full = self.mask_img[side_name] # for model
            mask_img_full = np.where(mask_img_full > 127, 255, 0).astype(np.uint8)
            # seamless mask ---------------------------------------------------------
            self.mask_img_seamless[side_name] = self.mask_img[side_name] # for model
            # --------------- transparent mask for model ---------------
            # Generate a transparent mask for deep fill models using the binarized, cropped mask
            self.df_mask[side_name] = gen_tranparent_mask(mask_img_crop)
            # If the generated mask has an alpha channel (4 channels), convert it to grayscale
            if self.df_mask[side_name].shape[-1] == 4:  # Check if the mask has an alpha channel
                self.df_mask[side_name] = cv2.cvtColor(self.df_mask[side_name], cv2.COLOR_RGBA2GRAY)
            # Add a channel dimension to the mask (shape: H x W x 1)
            self.df_mask[side_name] = np.expand_dims(self.df_mask[side_name], 2)
            # Add a channel dimension to the mask (shape: 1 x H x W x 1), convert to float32
            self.df_mask[side_name] = np.expand_dims(self.df_mask[side_name], 0).astype(np.float32)
            # Normalized to [0,1]
            self.df_mask[side_name] = (self.df_mask[side_name]/255.0).astype(np.float32)
        # project to equirectangular (get mask image) -----------------------------------------------------
        self.mask_pano_img = c2e(list(self.mask_img.values()), h=self.img_pano_h, w=self.img_pano_w, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]

    def set_segMask(self,seg_mask:dict) -> None:
        """
        Processes and sets segmentation masks for each side of a 3D model or image.
        Args:
            seg_mask (dict): A dictionary-like object containing segmentation masks for each side.
                                Each entry should be accessible by side name 
                                ["cube_front","cube_right","cube_back","cube_left","cube_top","cube_bottom"]
                                The value for each side should be a mask array or None.
        Workflow:
            - Loads and binarizes each mask, or creates a blank mask if None is provided.
            - Updates internal state to indicate which sides have valid masks.
            - Preprocesses masks by resizing/padding to match the original input size.
            - Generates seamless and transparent masks for further processing.
            - Ensures masks are in the correct format (shape and type) for model input.
            - Converts masks to grayscale if they have an alpha channel.
            - Normalizes masks to the [0, 1] range and adds necessary channel dimensions.
            - Combines all processed masks into a panorama mask for 360-degree processing.
        Side Effects:
            Updates the following instance attributes:
                - self.have_mask: Dict indicating presence of mask for each side.
                - self.in_mask: Dict of processed masks for model input.
                - self.mask_img: Dict of preprocessed masks matching input size.
                - self.mask_img_seamless: Dict of seamless masks for each side.
                - self.df_mask: Dict of transparent masks for deep fill models.
                - self.mask_pano_img: Panorama mask combining all sides.
        Note:
            Requires external functions: binary_change, up_downscale, gen_tranparent_mask, c2e.
            Assumes self.model_input, self.img_top_w, self.img_top_h, self.type_process_scale,
            self.img_pano_h, and self.img_pano_w are properly initialized.
        """

        for count, side_name in enumerate(seg_mask):
            # load mask ----------------------------
            if(seg_mask[side_name] is not None):
                mask = np.array(seg_mask[side_name]) # grayscale to ensure image is binary
                self.have_mask[side_name] = True
            else:
                mask = np.full((self.model_input[0],self.model_input[1],3), 0.0, np.float32)
                self.have_mask[side_name] = False
            mask_ori = binary_change(mask[:]) # Binarize
            mask_bw = cv2.bitwise_not(mask_ori[:]) # reversed mask to wanted black px (0)
            self.in_mask[side_name] = mask_bw # set for p2p model
            # preprocess mask -----------------------------------------------
            # pad image (upscale) to match original inpaut size | resize (upscale) to match original input size
            pad_mask = up_downscale((mask_ori/255.0).astype(np.float32),(self.img_top_w,self.img_top_h),'upscale',self.type_process_scale)
            self.mask_img[side_name] = pad_mask # full original input image 
            # seamless ---------------------------------------------------------
            self.mask_img_seamless[side_name] = (pad_mask[:]*255.0).astype(np.uint8)
            # --------------- transparent mask for model ---------------
            # Generate a transparent mask for deep fill models using the binarized, cropped mask
            self.df_mask[side_name] = gen_tranparent_mask(mask_ori)
            # If the generated mask has an alpha channel (4 channels), convert it to grayscale
            if self.df_mask[side_name].shape[-1] == 4:  # Check if the mask has an alpha channel
                self.df_mask[side_name] = cv2.cvtColor(self.df_mask[side_name], cv2.COLOR_RGBA2GRAY)
            # Add a channel dimension to the mask (shape: H x W x 1)
            self.df_mask[side_name] = np.expand_dims(self.df_mask[side_name], 2)
            # Add a channel dimension to the mask (shape: 1 x H x W x 1), convert to float32
            self.df_mask[side_name] = np.expand_dims(self.df_mask[side_name], 0).astype(np.float32)
            # Normalized to [0,1]
            self.df_mask[side_name] = (self.df_mask[side_name]/255.0).astype(np.float32)
        # ----------- change to panorama mask ---------------
        self.mask_pano_img = c2e(list(self.mask_img.values()), h=self.img_pano_h, w=self.img_pano_w, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        self.mask_pano_img = self.mask_pano_img*255.0

    def per_to_equi(self,car_path: Union[str,np.ndarray],per_img:np.ndarray) -> np.ndarray:
        """
        Converts a perspective image to an equirectangular (panoramic) format and applies inpainting.
        Args:
            car_path (Union[str, np.ndarray]): Path to the car panorama image as a string, or the image itself as a NumPy array.
            per_img (np.ndarray): The perspective image to be converted and inpainted.
        Returns:
            np.ndarray: The inpainted equirectangular (panoramic) image.
        Notes:
            - If `car_path` is a string, the image is loaded and preprocessed to the required panoramic size.
            - Inpainting is performed using the provided mask and the car panorama image.
        """
        if isinstance(car_path, str):
            car_pano_img = preprocess_img(car_path,img_size=(self.img_pano_w,self.img_pano_h))
        else:
            car_pano_img = car_path
        inpaint_pano_img = per_img.astype(np.uint8)
        inpaint_fill = inpaint_pano(inpaint_pano_img,self.mask_pano_img,car_pano_img)
        return inpaint_fill

    def overlay(self,pic_path: Union[str,np.ndarray],target_coor:int) -> np.ndarray:
        """
        Applies inpainting and overlay operations to a panoramic image or image array, returning the final composited panorama.
        Args:
            pic_path (Union[str, np.ndarray]): Path to the input panoramic image or a numpy array representing the image.
            target_coor (int): The target coordinate index used to construct the image filename if pic_path is a string.
        Returns:
            np.ndarray: The final composited panoramic image as a numpy array.
        Workflow:
            1. Loads and preprocesses the input panoramic image.
            2. Projects the equirectangular image to a cube map (six faces).
            3. For each cube face with a mask:
                - Downscales and preprocesses the face image.
                - Applies the selected inpainting model (pix2pix or deepfill).
                - Post-processes and upscales the inpainted result.
                - Blends the inpainted region seamlessly with the original face.
            4. For faces without a mask, uses a blank image.
            5. Reconstructs the panorama from the processed cube faces.
            6. Overlays the inpainted panorama onto the original input.
        Raises:
            ValueError: If an unsupported model type is specified.
        """

        output_cube = []
        # preprocess img ----------------------------------------------------
        all_time = 0.0
        if isinstance(pic_path, str):
            car_pano_img = pic_path + "img_"+str(target_coor).rjust(6, '0')+".jpg"
            # load image with preprocess
            car_pano_img = preprocess_img(car_pano_img,img_size=(self.img_pano_w,self.img_pano_h))
        else:
            car_pano_img = (np.array(pic_path*255)).astype(np.uint8)
        # project to perspective (get car image) ------------------------------
        image_cube = e2c(car_pano_img, face_w=1000, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        # inpainting & overlay process -------------------------------------------------------- 
        for count, side_name in enumerate(self.have_mask):
            # check if that cube side have mask then process to inpaint said side
            if(self.have_mask[side_name]):
                # crop image (downscale) to match model input size  | resize (downscale) to match model input size
                per_img = up_downscale(image_cube[count],(self.model_input[0], self.model_input[1]),'downscale',self.type_process_scale) # resize (downscale) to match model input size
                car_img_side = preprocess_img(per_img,img_size=(self.model_input[0],self.model_input[1])) # guarantee data range and image size to match model input image size
                car_img_side = car_img_side.astype('float32') # cast to float32 type
                if(self.model_type == "p2p"):
                    # generate input format for model
                    testgen = createAugment(np.array([car_img_side]), np.array([car_img_side]),np.array([self.in_mask[side_name]]),batch_size=1,dim=(self.model_input[0],self.model_input[1]), shuffle=False, random_mask=False)
                    [masked_image, mask], car_img = testgen[0]
                    # pix2pix ------------------------------------
                    car_img = (car_img - 0.5)/0.5 # normalize [-1,1]
                    masked_image = (masked_image - 0.5)/0.5 # normalize [-1,1]
                    inputs = masked_image # pix2pix
                    predict_image = self.generator(inputs, training=True)
                    predict_image_norm = ((predict_image[0]* 0.5) + 0.5).numpy()# de-normalize [0,1]
                    masked_image = (masked_image* 0.5) + 0.5 # de-normalize [0,1]
                    car_img = (car_img* 0.5) + 0.5 # de-normalize [0,1]
                    # ------------------------------------------------
                elif(self.model_type == "df"):
                    image = np.expand_dims(car_img_side, 0).astype(np.float32) # generate input format for model
                    image = image / 255.0 # normalize input image
                    masked_image = image*(1. - self.df_mask[side_name]) # masked to remove target area
                    x1, x2, offset_flow = self.generator(masked_image, self.df_mask[side_name], training=False) # inpainting
                    batch_predicted = x2
                    # result = batch_predicted  * self.df_mask[side_name] + masked_image * (1. - self.df_mask[side_name]) # crude overlay 
                    predict_image_norm = np.array(batch_predicted[0]) # put result into variable
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")
                predict_image_norm = preprocess_img(predict_image_norm,img_size=(self.model_input[0],self.model_input[1])) # guarantee data range and image size to match model input
                # padding (upscale) image to match original input size | resize (upscale) image to match original input size
                predict_image_norm = up_downscale(predict_image_norm,(self.img_top_w, self.img_top_h),'upscale',self.type_process_scale) # resize image to match real input size
                # predict_image_norm = ((1 - mask_img_pano) * per_img_1000) + (mask_img_pano * predict_image_norm) # crude overlay
                predict_image_norm = self.seamless((predict_image_norm*255.0).astype(np.uint8),(image_cube[count]).astype(np.uint8),self.mask_img_seamless[side_name]) # soft overlay
                output_cube.append(predict_image_norm) # append inpainting result
            # if not have mask for inpainting append blank image (all black)
            else:
                output_cube.append(self.blank_side)
        # overlay panorama ---------------------------------------
        # cube map -> equirectangular projection
        cube_pano_img = c2e(output_cube, h=self.img_pano_h, w=self.img_pano_w, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom] # pane inpaint cube to panorama
        final_pano_output = self.per_to_equi(car_pano_img,cube_pano_img) #overlaying
        return final_pano_output
    


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

        self.mask_img = cv2.resize(cv2.imread(find_file_path(self.mask)),(self.img_top_h,self.img_top_w)) # for model
        # project to equirectangular (get mask image)
        equ = Equirectangular(self.mask_img, 167, self.coor, -90,'img')    
        mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w) 
        self.mask_pano_img = mask_pano_img.astype(np.uint8)
        print('Complete setting '+config_name)

    def seamless(self,source:np.ndarray, target:np.ndarray, mask:np.ndarray) -> np.ndarray:
        """
        Performs seamless cloning of a source image onto a target image using a specified mask.
        This method computes the bounding rectangle of the mask to determine the center point for seamless cloning.
        It then blends the source image into the target image at the computed center using OpenCV's seamlessClone function.
        Args:
            source (np.ndarray): The source image to be cloned onto the target.
            target (np.ndarray): The target image where the source will be blended.
            mask (np.ndarray): The mask defining the region of the source image to be cloned.
        Returns:
            np.ndarray: The resulting image after seamless cloning.
        """

        x, y, w, h = cv2.boundingRect(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

        # Compute center of the bounding box
        center = (x + w // 2, y + h // 2)
        # Get seamless center from config data setting
        # (h, w) = (self.seamless_center_h, self.seamless_center_w)
        # center = (h,w)
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

    def equi_to_per(self,car_pano_img:np.ndarray) -> np.ndarray:
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
        # car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
        equ = Perspective(car_pano_img)    # Load equirectangular image
        per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)  # Specify parameters(FOV, theta, phi, height, width)
        return per_img_1000
    
    def save_pano(self,inpaint_fill:np.ndarray, save_path:str, count:str) -> None:
        """
        Saves an inpainted panorama image to the specified path.
        Args:
            inpaint_fill (np.ndarray): The inpainted image as a NumPy array with values in [0, 1].
            save_path (str): The directory path where the image will be saved.
            count (str): A string or number used to generate a zero-padded filename.
        Returns:
            None
        """
        impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
        impainted_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg")

    def per_to_equi(self,car_pano_img:np.ndarray, per_img:np.ndarray) -> np.ndarray:
        """
        Projects a perspective image onto an equirectangular panorama and performs inpainting.
        Args:
            car_pano_img (np.ndarray): The original panoramic image of the car.
            per_img (np.ndarray): The perspective image to be projected onto the panorama.
        Returns:
            np.ndarray: The inpainted panoramic image after projecting the perspective image and applying inpainting.
        """
        equ = Equirectangular(per_img,167, self.coor, -90)    
        inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
        inpaint_fill = inpaint_pano(inpaint_pano_img,self.mask_pano_img,car_pano_img)
        return inpaint_fill

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
        source_input = preprocess_img(main_path + "img_"+str(source_path).rjust(6, '0')+".jpg",(self.img_pano_w, self.img_pano_h)) 
        target_input = preprocess_img(main_path + "img_"+str(target_path).rjust(6, '0')+".jpg",(self.img_pano_w, self.img_pano_h)) 
        source = self.equi_to_per(source_input)
        target = self.equi_to_per(target_input)
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
        result = self.seamless(source,target,self.mask_img)
        inpaint_pano = self.per_to_equi(target_input,result)
        return inpaint_pano

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