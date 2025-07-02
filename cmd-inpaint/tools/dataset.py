import cv2
import numpy as np
from tensorflow import keras
from tqdm import tqdm
from typing import Union

from .process_img import preprocess_img, move_img, flip_img, zoom_img


## Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
class createAugment(keras.utils.Sequence):
    """
    Keras data generator for image augmentation and inpainting tasks.
    This generator yields batches of masked images, corresponding masks, and original images for training models that require masked input (e.g., partial convolutional networks for image inpainting).
    Args:
        X (np.ndarray): Array of input images.
        y (np.ndarray): Array of target images (ground truth).
        mask_ds (np.ndarray, optional): Array of masks to apply to images. If None, masks are generated dynamically.
        batch_size (int, optional): Number of samples per batch. Default is 8.
        dim (tuple, optional): Dimensions of the images (height, width). Default is (32, 32).
        n_channels (int, optional): Number of image channels. Default is 3.
        shuffle (bool, optional): Whether to shuffle the data after each epoch. Default is True.
        random_mask (bool, optional): Whether to generate random masks (lines, circles, ellipses) for augmentation. Default is False.
        other_car_list (np.ndarray, optional): Array of additional images to use for mask augmentation (e.g., overlaying other objects).
    Methods:
        __len__(): Returns the number of batches per epoch.
        __getitem__(index): Generates one batch of data.
        on_epoch_end(): Updates indexes after each epoch (shuffling if enabled).
        __data_generation(idxs): Generates data for a batch, including masked images, masks, and ground truth.
        __createMask(img, idx): Creates a mask for a given image, optionally applying random augmentations or overlaying objects.
    Returns:
        A tuple ([masked_images, masks], original_images) for each batch, where:
            - masked_images: Batch of input images with masks applied.
            - masks: Batch of binary masks used for masking.
            - original_images: Batch of original (ground truth) images.
    """

    """Generates data for Keras"""
    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            mask_ds: np.ndarray = None,
            batch_size: int = 8,
            dim: tuple = (32, 32),
            n_channels: int = 3,
            shuffle: bool = True,
            random_mask: bool = False,
            other_car_list: np.ndarray = None
    ) -> None:
        'Initialization'
        self.batch_size = batch_size
        self.X = X/255.0
        self.y = y/255.0
        if mask_ds is not None:
            self.mask_ds = mask_ds/255.0
        else:
            self.mask_ds = mask_ds
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.random_mask = random_mask
        if other_car_list is not None:
            self.random_car = True
            self.other_car_list = other_car_list/255.0
        else:
            self.random_car = False
        self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index: int) -> tuple:
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X_input_ds, y_output = self.__data_generation(indexes)
        return X_input_ds, y_output

    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idxs) -> tuple:
        """
        Generates a batch of data for training or inference.
        Given a list of indices, this method prepares and returns a batch of masked images,
        corresponding binary masks, and original images. The masked images and masks are
        generated using the `__createMask` method. The output is structured to support
        models that require both masked images and masks as input (e.g., partial convolution networks).
        Args:
            idxs (list or array-like): List of indices specifying which samples to include in the batch.
        Returns:
            tuple: A tuple containing:
                - [Masked_images, Mask_batch] (list of np.ndarray): 
                    Masked_images is a batch of images with masks applied.
                    Mask_batch is a batch of corresponding binary masks.
                - y_batch (np.ndarray): Batch of original, unmasked images (ground truth).
        """

        # Masked_images is a matrix of masked images used as input
        Masked_images = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Masked image
        # Mask_batch is a matrix of binary mask_ds used as input
        Mask_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Binary Mask_dsmask_ds
        # y_batch is a matrix of original images used for computing error from reconstructed image
        y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Original image


        ## Iterate through random indexes
        for i, idx in enumerate(idxs):
            image_copy = self.X[idx].copy()

            ## Get mask associated to that image
            masked_image, mask = self.__createMask(image_copy,idx)

            Masked_images[i,] = masked_image
            Mask_batch[i,] = mask
            y_batch[i] = self.y[idx]

        ## Return mask as well because partial convolution require the same.
        return [Masked_images, Mask_batch], y_batch

    def __createMask(self, img:np.ndarray, idx:int) -> tuple:
        """
        Generates a mask for the given image and index, applying various random augmentations if enabled.
        Parameters:
            img (np.ndarray): The input image to be masked.
            idx (int): The index used to retrieve a mask from the mask dataset, if available.
        Returns:
            tuple:
                masked_image (np.ndarray): The resulting image after applying the generated mask.
                mask (np.ndarray): The mask matrix applied to the image.
        Behavior:
            - If a mask dataset (`self.mask_ds`) is provided, retrieves the mask at the given index.
            - If no mask dataset is provided, creates a white mask of the same dimensions as the image.
            - If `self.random_car` is enabled, randomly selects and augments a car image from `self.other_car_list` and adds it to the mask.
            - If `self.random_mask` is enabled, draws random lines, circles, and ellipses on the mask to create random patterns.
            - The mask is applied to the input image, setting masked regions to white.
        Raises:
            Exception: If the mask dimensions are less than 64x64 when `self.random_mask` is enabled.
        """

        ## Prepare masking matrix
        if(self.mask_ds is not None):
            mask = self.mask_ds[idx]
        elif(self.mask_ds is None):
            mask = np.full((self.dim[0],self.dim[1],3), 1.0, np.float32) ## White background
        if(self.random_car):
            for _ in range(np.random.randint(1, 3)):
                other_car = self.other_car_list[np.random.randint(0, len(self.other_car_list)-1)]
            if(bool(np.random.randint(0, 1))): # move_img
                other_car = move_img(other_car,np.random.randint(-50, 50),np.random.randint(-50, 50))
            if(bool(np.random.randint(0, 1))): # zoom
                other_car = zoom_img(other_car, zoom_factor=np.random.uniform(0.5,2.0))
            if(bool(np.random.randint(0, 1))): # flip
                other_car = flip_img(other_car)
            mask = mask + other_car
            mask = (mask > 1).astype(np.float32)
            # mask = mask - 1.0
        # Set size scale
        if(self.random_mask):
            size = int((self.dim[0] + self.dim[1]) * 0.03)
            if self.dim[0] < 64 or self.dim[1] < 64:
                raise Exception("Width and Height of mask must be at least 64!")
            # Draw random lines
            for _ in range(np.random.randint(1, 10)):
                # Get random x locations to start line
                x1, x2 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[1])
                # Get random y locations to start line
                y1, y2 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[1])
                # Get random thickness of the line drawn
                thickness = np.random.randint(1, 3)
                # Draw black line on the white mask
                cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),thickness)
            # Draw random circles
            for _ in range(np.random.randint(1, 20)):
                x1, y1 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[1])
                radius = np.random.randint(3, size)
                cv2.circle(mask,(x1,y1),radius,(0,0,0), -1)
            # Draw random ellipses
            for _ in range(np.random.randint(1, 20)):
                x1, y1 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[1])
                s1, s2 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[1])
                a1, a2, a3 = np.random.randint(3, 180), np.random.randint(3, 180), np.random.randint(3, 180)
                thickness = np.random.randint(3, size)
                cv2.ellipse(mask, (x1,y1), (s1,s2), a1, a2, a3,(0,0,0), thickness)

        ## Mask the image
        masked_image = mask * img + (1 - mask) * 1.0

        return masked_image, mask

class Dataset:
    """
    A class for loading and preprocessing datasets consisting of input images, masks, and labels.
    Attributes:
        main_dir (str): The main directory path where the dataset files are located.
        input_dir (str): Path to the file containing the list of input image filenames.
        mask_dir (str): Path to the file containing the list of mask image filenames. Can be None.
        label_dir (str): Path to the file containing the list of label image filenames.
        image_size (tuple or list): Desired size (width, height) to which images will be resized.
        input_ds (list): List to store preprocessed input images.
        label_ds (list): List to store preprocessed label images.
        mask_ds (list): List to store preprocessed mask images.
    Methods:
        __init__(main_dir, input_dir, mask_dir, label_dir, image_size=(128,128)):
            Initializes the Dataset object with directory paths and image size.
        process_data():
            Loads and preprocesses images from the specified directories.
            Returns:
                tuple: (input_ds, label_ds, mask_ds) as numpy arrays of float32 type.
    """

    def __init__(self,main_dir:str,input_dir:str,mask_dir:str,label_dir:str,image_size:Union[tuple,list]=(128,128)):
        self.main_dir = main_dir
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.input_ds = []
        self.label_ds = []
        self.mask_ds = []

    
    def process_data(self):
        with open(self.input_dir) as f:
            for count, line in enumerate(tqdm(f)):
                # print(line.strip())
                img = preprocess_img(self.main_dir+line.strip() ,self.image_size)
                self.input_ds.append(img)
        self.input_ds = np.array(self.input_ds).astype('float32')

        if self.mask_dir is not None:
            with open(self.mask_dir) as f:
                for count, line in enumerate(tqdm(f)):
                    # print(line.strip())
                    img = preprocess_img(self.main_dir+ line.strip(),self.image_size)
                    img = cv2.bitwise_not(img)
                    self.mask_ds.append(img)
            self.mask_ds = np.array(self.mask_ds).astype('float32')
        else:
            self.mask_ds = None

        with open(self.label_dir) as f:
            for count, line in enumerate(tqdm(f)):
                # print(line.strip())
                img = preprocess_img(self.main_dir + line.strip() ,self.image_size)
                self.label_ds.append(img)
        self.label_ds = np.array(self.label_ds).astype('float32')

        return(self.input_ds,self.label_ds,self.mask_ds)
    
def prepare_other_car_mask(main_dir:str,image_size:Union[tuple,list]) -> np.ndarray:
    """
    Generates a mask array for 'other cars' by reading image paths from a configuration file,
    preprocessing each image, inverting its pixel values, and aggregating the results into a NumPy array.
    Args:
        main_dir (str): The base directory path where the dataset and configuration file are located.
        image_size (Union[tuple, list]): The desired size (height, width) to which each image will be resized.
    Returns:
        np.ndarray: A NumPy array of shape (N, H, W, C) containing the processed and inverted images,
                  where N is the number of images, and (H, W, C) corresponds to the specified image size and channels.
    """

    other_car_list = []
    with open(main_dir + 'car_ds/train_test_config/random_car.txt') as f:
      for count, line in enumerate(f):
          img = preprocess_img(line.strip(),image_size)
          img = cv2.bitwise_not(img)
          other_car_list.append(img)
    other_car_list = np.array(other_car_list).astype('float32')

    return other_car_list