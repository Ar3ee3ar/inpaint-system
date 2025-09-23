# from https://github.com/timy90022/Perspective-and-Equirectangular/tree/master

import cv2
import numpy as np

class Perspective:
    """
    Generates a perspective projection from an equirectangular (panoramic) image.
    Args:
        FOV (float): Field of view in degrees for the output perspective image.
        THETA (float): Horizontal rotation angle (left/right) in degrees.
        PHI (float): Vertical rotation angle (up/down) in degrees.
        height (int): Height of the output perspective image.
        width (int): Width of the output perspective image.
    Returns:
        numpy.ndarray: The perspective-projected image as a NumPy array.
    Notes:
        - The function computes the perspective transformation by mapping the specified FOV and orientation (THETA, PHI)
          onto the equirectangular source image.
        - Uses OpenCV for remapping and rotation matrix calculations.
        - Handles border wrapping for seamless panoramic images.
    """

    def __init__(self, img:np.ndarray) -> None:
        """
        Initialize data to Generates a perspective projection from an equirectangular (panoramic) image.
        Args:
            img (np.ndarray): Numpy equirectangular (panoramic) image
        """
        self._img = img
        
        [self._height, self._width, _] = self._img.shape
        # print(self._img.shape)
    

    def GetPerspective(self, FOV:float, THETA:float, PHI:float, height:int, width:int) -> np.ndarray:
        """
        Generates a perspective view from an equirectangular panorama image.
        Args:
            FOV (float): Field of view in degrees for the output perspective image.
            THETA (float): Horizontal rotation angle (left/right) in degrees.
            PHI (float): Vertical rotation angle (up/down) in degrees.
            height (int): Height of the output perspective image.
            width (int): Width of the output perspective image.
        Returns:
            numpy.ndarray: The perspective image generated from the equirectangular panorama.
        Notes:
            - The function computes the mapping from the perspective view to the equirectangular panorama using rotation matrices.
            - The output image is generated using OpenCV's remap function with cubic interpolation and border wrapping.
            - THETA and PHI specify the center direction of the perspective view in spherical coordinates.
        """
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        
        # max x,y (จาก perspective ครึ่งรูป)
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0
        
        # คิด FOV
        wFOV = FOV
        hFOV = float(height) / width * wFOV

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))

        # สร้าง xyz จาก FOV ที่ต้องการ
        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len,width), [height,1])
        z_map = -np.tile(np.linspace(-h_len, h_len,height), [width,1]).T # transpose
        
        # หาค่า r
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        
        xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)
        
        # rotation matrix
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA)) # rotation around y-axis (use theta) 
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI)) # rotation z-axis (use phi)
        xyz = xyz.reshape([height * width, 3]).T # transpose of xyz
        # use rotation matrix with picture (cartesian coordinates - xyz)
        xyz = np.dot(R1, xyz)  
        xyz = np.dot(R2, xyz).T
        
        # uv mapping
        lat = np.arcsin(xyz[:, 2]) # phi (v)
        lon = np.arctan2(xyz[:, 1] , xyz[:, 0]) # theta (u)
        lon = lon.reshape([height, width]) / np.pi * 180 # เปลี่ยนเป็น degree แล้ว
        lat = -lat.reshape([height, width]) / np.pi * 180 # เปลี่ยนเป็น degree แล้ว
        lon = ((lon / 180) * equ_cx + 0.5) + equ_cx -0.5 # /180 เพราะเป็นพิกัดแนว y (-180,180) # opencv 0.5 shift
        lat = ((lat / 90)  * equ_cy + 0.5) + equ_cy -0.5# /90 เพราะเป็นพิกัดแนว x (-90,90) # opencv 0.5 shift

        # mapping image to desired coordinates 
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp
    
class Equirectangular:
    """
    A class for processing and transforming equirectangular panoramic images.
    This class provides methods to convert an input image (or mask) into an equirectangular projection
    with specified field of view (FOV) and rotation angles (THETA, PHI). It supports both image and mask
    types, and can generate perspective views from the equirectangular panorama.
    Attributes:
        _img (np.ndarray): The input image or mask, possibly inverted if type_img is 'mask'.
        _height (int): Height of the input image.
        _width (int): Width of the input image.
        wFOV (float): Horizontal field of view in degrees.
        THETA (float): Rotation angle around the z-axis (left/right) in degrees.
        PHI (float): Rotation angle around the y-axis (up/down) in degrees.
        hFOV (float): Computed vertical field of view in degrees.
        w_len (float): Tangent of half the horizontal FOV in radians.
        h_len (float): Tangent of half the vertical FOV in radians.
    Methods:
        __init__(img_name, FOV, THETA, PHI, type_img='img'):
            Initializes the Equirectangular object with the given image, FOV, rotation angles, and type.
        GetEquirec(height, width):
            Generates an equirectangular projection of the image with the specified output height and width.
            Applies the specified FOV and rotation angles, and returns the perspective image and mask.
            Args:
                height (int): The height of the output equirectangular image.
                width (int): The width of the output equirectangular image.
            Returns:
                tuple: A tuple containing:
                    - persp (np.ndarray): The perspective-transformed image.
                    - mask (np.ndarray): The mask indicating valid regions in the output image.
    """

    def __init__(self, img:np.ndarray , FOV:float, THETA:float, PHI:float ,type_img:str='img'):
        """
        Initializes the Equirectangular object with the given image, FOV, rotation angles, and type.
        """
        self._img = img
        if(type_img == 'mask'):
            self._img = cv2.bitwise_not(self._img)
        [self._height, self._width, _] = self._img.shape
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
#         self.hFOV = float(self._height) / self._width * FOV
        self.hFOV = np.rad2deg(2 * np.arctan(self._height * np.tan(np.radians(FOV/2)) / self._width))
        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

    

    def GetEquirec(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates an equirectangular projection of the image with the specified output height and width.
            Applies the specified FOV and rotation angles, and returns the perspective image and mask.
            Args:
                height (int): The height of the output equirectangular image.
                width (int): The width of the output equirectangular image.
            Returns:
                tuple: A tuple containing:
                    - persp (np.ndarray): The perspective-transformed image.
                    - mask (np.ndarray): The mask indicating valid regions in the output image.
        """
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        
        # meshgrid (สร้าง grid ที่มีจุดกระจายอยู่เป็นคู่ตน. x,y ทุกจุด) ทั้งรูปภาพ y(height) -> (-90,90), x(width) -> (-180,180) 
        x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
        
        # 
        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map,y_map,z_map),axis=2)
        
        # rotation matrix
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA)) # rotation around z-axis
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI)) # rotation around y-axis
        
        # inverse matrix
        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)
        
        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height , width, 3])
        inverse_mask = np.where(xyz[:,:,0]>0,1,0) # จำกัดเขตที่จะสร้างภาพ equirectangular 

        xyz[:,:] = xyz[:,:]/np.repeat(xyz[:,:,0][:, :, np.newaxis], 3, axis=2)
        
        # w_len, h_len -> field of view
        # หาตน. ที่มี xyz ตรงกันระหว่าง coordinates เป็นผืนผ้า equirec กับ 
        # lat lon - พิกัดบนล่าง
        lon_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(xyz[:,:,1]+self.w_len)/2/self.w_len*self._width,0)
        lat_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height,0)
        mask = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),1,0)
        
        # map pixel ตาม lat lon coordinates
        persp = cv2.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) # old : cv2.INTER_CUBIC
        
        # เอาเฉพาะช่วงล่าง
        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        persp = persp * mask
        
        return persp , mask
    
class MultiPerspective:
    """
    MultiPerspective class for merging multiple perspective images into a single equirectangular panorama.
    Attributes:
        img_array (list): List of input images or image file paths.
        F_T_P_array (list): List of [FOV, Theta, Phi] parameter lists for each image.
    Methods:
        __init__(img_array, F_T_P_array):
            Initializes the MultiPerspective object with arrays of images and their corresponding FOV, Theta, and Phi parameters.
        GetEquirec(height, width):
            Generates a merged equirectangular image of the specified height and width by projecting each input image using its parameters and averaging overlapping regions.
            Args:
                height (int): Height of the output equirectangular image.
                width (int): Width of the output equirectangular image.
            Returns:
                numpy.ndarray: The merged equirectangular image as a NumPy array.
    """

    def __init__(self, img_array:list , F_T_P_array:list ):
        """
        Initializes the MultiPerspective object with arrays of images and their corresponding FOV, Theta, and Phi parameters.
        """
        assert len(img_array)==len(F_T_P_array)
        
        self.img_array = img_array
        self.F_T_P_array = F_T_P_array
    

    def GetEquirec(self, height:int, width:int):
        """
        Generates a merged equirectangular image of the specified height and width by projecting each input image using its parameters and averaging overlapping regions.
            Args:
                height (int): Height of the output equirectangular image.
                width (int): Width of the output equirectangular image.
            Returns:
                numpy.ndarray: The merged equirectangular image as a NumPy array.
        """
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        merge_image = np.zeros((height,width,3))
        merge_mask = np.zeros((height,width,3))

        for img_dir,[F,T,P] in zip (self.img_array,self.F_T_P_array):
            equ = Equirectangular(img_dir,F,T,P)        # Load equirectangular image
            img , mask = equ.GetEquirec(height,width)   # Specify parameters(FOV, theta, phi, height, width)
            merge_image += img
            merge_mask +=mask
        merge_mask = np.where(merge_mask==0,1,merge_mask)
        merge_image = (np.divide(merge_image,merge_mask))
        
        return merge_image
    
def panorama2cube(image:np.ndarray, cube_size:int = 1000) -> list:
    """
    Converts an equirectangular panorama image to six cube face images.
    Args:
        image (np.ndarray): The input equirectangular panorama image.
        cube_size (int, optional): The size (width and height) of each cube face image. Defaults to 1000.
    Returns:
        list: A list of six images corresponding to the cube faces in the following order:
            [front, right, back, left, top, bottom].
    Note:
        Requires the Perspective class with a GetPerspective method for extracting cube faces.
    """

    equ = Perspective(image)    # Load equirectangular image
    #
    # FOV unit is degree
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension
    #

    f_img = equ.GetPerspective(90, 0, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
    r_img = equ.GetPerspective(90, 90, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
    back_img = equ.GetPerspective(90, 180, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
    l_img = equ.GetPerspective(90, 270, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
    t_img = equ.GetPerspective(90, 0, 90, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
    bot_img = equ.GetPerspective(90, 0, -90, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)

    return [f_img, r_img, back_img, l_img, t_img, bot_img]

def cube2panorama(image_list:list, width:int, height:int) -> np.ndarray:
    """
    Converts a list of six cube face images into a single equirectangular panorama image.
    Args:
        image_list (list): A list of six images representing the cube faces in the following order:
            [front, right, back, left, top, bottom].
        width (int): The width of the output panorama image.
        height (int): The height of the output panorama image.
    Returns:
        numpy.ndarray: The resulting equirectangular panorama image.
    Raises:
        IndexError: If image_list does not contain exactly six images.
    """

    front = image_list[0]
    right = image_list[1]
    back = image_list[2]
    left = image_list[3]
    top = image_list[4]
    bottom = image_list[5]

    # this can turn cube to panorama
    per = MultiPerspective([front,right,back,left,top,bottom],
                            [[90, 0, 0],[90, 90, 0],[90, 180, 0],
                            [90, 270, 0],[90, 0, 90],[90, 0, -90]])    
    
    
    img = per.GetEquirec(height,width)  
    return img
    
def inpaint_pano(inpaint_pano_img:np.ndarray, mask_pano_img:np.ndarray, car_pano_img:np.ndarray) -> np.ndarray:
    """
    Inpaints a panoramic image by blending an inpainted region with the original image using a mask.
    Args:
        inpaint_pano_img (np.ndarray): The inpainted panoramic image as a NumPy array, with pixel values in [0, 255].
        mask_pano_img (np.ndarray): The mask image as a NumPy array, with pixel values in [0, 255]. Non-zero values indicate regions to use from the inpainted image.
        car_pano_img (np.ndarray): The original panoramic image as a NumPy array, with pixel values in [0, 255].
    Returns:
        np.ndarray: The blended panoramic image as a NumPy array with pixel values in [0, 1], where the inpainted and original images are combined according to the mask.
    """
    # cast to float32
    inpaint_pano_img = (inpaint_pano_img/255.0).astype(np.float32)
    mask_pano_img = (mask_pano_img/255.0).astype(np.float32) # for rect_mask
    # mask_pano_img[mask_pano_img != 0] = 1 # for car_mask
    car_pano_img = (car_pano_img/255.0).astype(np.float32)
    # inpaint_fill_pano = np.where(mask_pano_img>0.5, inpaint_pano_img,  car_pano_img) # hard overlay (have sharp border between outter and inner image)
    # (select only inpaint area) + (reversed mask to make outter area into non-zero pixel to get only non-inpainted area)
    inpaint_fill_pano = ((mask_pano_img) * inpaint_pano_img) + ((1-mask_pano_img) * car_pano_img) # soft overlay (mixed between outter and inner image)
    return inpaint_fill_pano

# def inpaint_random(inpaint_pano_img,mask,car_pano_img):
