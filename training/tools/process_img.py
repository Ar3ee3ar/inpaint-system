import cv2
import numpy as np
import tensorflow as tf
from typing import Union
# import tensorflow_io as tfio

tf.config.run_functions_eagerly(True)

def preprocess_img (path_img: Union[str,np.ndarray], img_size=(512,512)) -> np.ndarray:
  """
  Preprocesses an image by loading it from a file path or using an existing image array,
  converting it to RGB format if loaded from a file, and resizing it to the specified size.
  Args:
    path_img (str or numpy.ndarray): Path to the image file or an image array.
    img_size (tuple, optional): Desired output size as (width, height). Defaults to (512,512).
  Returns:
    numpy.ndarray: The preprocessed image as a NumPy array in RGB format and resized to img_size.
  """
  if isinstance(path_img, str): # input 
    img = cv2.imread(path_img)
    if img is None:
      raise FileNotFoundError(f"Image file not found or could not be read: {path_img}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  else:
    img = path_img
  img = cv2.resize(img, img_size) # img_size = (w,h)
  return img

def preprocess_vgg(image:np.ndarray) -> tf.Tensor:
  """
  Preprocesses a batch of images for VGG-style models.
  Resizes each image in the input batch to 224x224 pixels using bilinear interpolation,
  and converts the batch to a TensorFlow tensor of type float32.
  Args:
    image (Iterable or np.ndarray): A batch of images, where each image is expected to be a NumPy array or tensor.
  Returns:
    tf.Tensor: A tensor containing the resized images with shape (batch_size, 224, 224, channels) and dtype float32.
  Notes:
    - Use for training Discriminator loss of Pix2Pix model
  """

  process_img = tf.convert_to_tensor([tf.image.resize(x, [224,224], method='bilinear') for x in image], dtype=tf.float32)
  # process_img = tfio.experimental.color.rgb_to_bgr(process_img)
  # process_img = process_img * 255
  return process_img

def crop_center(image:np.ndarray, new_img_size:Union[list,tuple]=(190,290)) -> np.ndarray: # new_width=190,new_height=290
  """
  Select image by cropping from center of image
  Args:
    image (np.ndarray): Input image
    new_img_size (list | tuple): wanted crop size of input image
  Return:
    np.ndarray: Input image after crop process that have size of new_img_size
  """
  center = image.shape
  new_width = new_img_size[0]
  new_height = new_img_size[1]
  x = center[1]/2 - new_width/2
  y = center[0]/2 - new_height/2

  crop_img = image[int(y):int(y+new_height), int(x):int(x+new_width)]
  return crop_img

def crop_center_position(image:np.ndarray, new_img_size:Union[list,tuple]=(190,290),move_x:int=0,move_y:int=0) -> np.ndarray: # new_width=190,new_height=290
  """
  Select image by cropping from that move from x pixel(width) and y pixel(height) center of image
  Args:
    image (np.ndarray): Input image
    new_img_size (list | tuple): wanted crop size of input image
    move_x (int): pixel that move from center in term of width
    move_y (int): pixel that move from center in term of height
  Return:
    np.ndarray: Input image after crop process that have size of new_img_size
  """
  center = image.shape
  new_width = new_img_size[0]
  new_height = new_img_size[1]
  x = (center[1]/2 - new_width/2) + move_x
  y = (center[0]/2 - new_height/2) + move_y

  crop_img = image[int(y):int(y+new_height), int(x):int(x+new_width)]
  return crop_img

def resize(img:np.ndarray, new_width:int, new_height:int) -> np.ndarray:
  """
  Resize image to match new_width and new_height
  Args:
    img (np.ndarray): Input numpy image
    new_width (int): New width size to match
    new_height (int): New height size to match
  Return:
    np.ndarray: Input after resize process that have size of (new_width, new_height)
  """
  resize_img = cv2.resize(img, [new_width,new_height], interpolation = cv2.INTER_AREA)
  return resize_img

def pad_images_to_same_size(images:np.ndarray, new_image_width:int = 1000,new_image_height:int=1000) -> np.ndarray:
    """
    Padding process by add black pixel (padding) until to made image to match new image size
    Args:
      images (np.ndarray): Input numpy image
      new_image_width (int): New width size to match
      new_image_height (int): New height size to match
    Return:
      np.ndarray: Input image after padding process that have size of (new_image_width, new_image_height)
    """
    old_image_height, old_image_width, channels = images.shape

    # create new image of desired size and color (blue) for padding
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.float32)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = images
    # plt.imshow(result)
    # plt.show()
    return result

def move_img(img:np.ndarray, tx:int, ty:int) -> np.ndarray:
    """
    Moving image in x-axis and y-axis (width and height)
    Args:
      img (np.ndarray): Input numpy image
      tx (int): move in width of image (x-axis)
      ty (int): move in height of image (y-axis)
    Return:
      np.ndarray: Input image after 
    Note:
      - Use for GPS inpainting process by moving img(t-1) that called related image to match position of img(t) that called target image
    """
    M = np.float32([[1,0,tx],[0,1,ty]])
    dst = cv2.warpAffine(img,M,(img.shape[0],img.shape[1]),borderValue=(1,1,1))
    return dst

def zoom_img(img:np.ndarray, zoom_factor:int=2) -> np.ndarray:
    """
    Zoom image process
    Args:
      img (np.ndarray): Input numpy image
      zoom_factor (int): zoom ratio
    Return:
      np.ndarray: Input image after zoom process
    Notes:
      - Use for data augmentation process in generate dataset (before training or during training)
    """
    return cv2.resize(img, (img.shape[0],img.shape[1]), fx=zoom_factor, fy=zoom_factor)

def flip_img(img:np.ndarray) -> np.ndarray:
    """
    Horizontal flipping image process (left to right or vice versa)
    Args:
      img (np.ndarray): Input numpy image
    Return:
      np.ndarray: Input image after flipping 
    """
    return cv2.flip(img, 1)

def binary_change(mask):
  binary_mask = np.where(mask > 127, 255, 0).astype(np.uint8)
  return binary_mask