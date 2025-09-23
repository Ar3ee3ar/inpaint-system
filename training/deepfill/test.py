from inpaint_model import InpaintGenerator
from PIL import Image
import yaml
import argparse

import cv2
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

def gen_tranparent_mask(src):
    # Convert image to image gray 
    print(src.shape)
    if(len(src.shape)==3):
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 
        
        # Applying thresholding technique 
        _, alpha = cv2.threshold(tmp, thresh=127, maxval=255, type=cv2.THRESH_BINARY) 
        
        # Using cv2.split() to split channels  
        # of coloured image 
        b, g, r = cv2.split(src) 
        
        # Making list of Red, Green, Blue 
        # Channels and alpha 
        rgba = [b, g, r, alpha] 
        
        # Using cv2.merge() to merge rgba 
        # into a coloured/multi-channeled image 
        dst = cv2.merge(rgba, 4) 
        # Writing and saving to a new image 
        #cv2.imwrite(save_path+count+"_gfg_white.png", dst)
    elif(len(src.shape)==2):
        # Convert to numpy array
        gray_np = np.array(src)

        # Create an RGBA image: black with alpha from grayscale
        rgba_np = np.zeros((gray_np.shape[0], gray_np.shape[1], 4), dtype=np.uint8)
        rgba_np[..., 3] = gray_np  # set alpha channel 
        dst = rgba_np
    plt.imshow(dst)
    plt.show()
    return dst

def binary_change(mask,save_path=''):
    # Apply threshold to make it binary (0 and 1)
    binary_mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    # print(binary_mask)
    # cv2.imwrite(save_path,binary_mask)
    plt.imshow(binary_mask)
    plt.show()
    return binary_mask


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')



if __name__ == "__main__":
    FLAGS = yaml.load(open('inpaint.yml', 'r'), Loader=yaml.FullLoader)
    args, unknown = parser.parse_known_args()

    image = np.array(Image.open(args.image).convert('RGB'))
    mask = np.array(Image.open(args.mask).convert('RGB')) # grayscale to ensure image is binary
    mask = gen_tranparent_mask(binary_change(mask))
    # Convert transparent mask to grayscale
    if mask.shape[-1] == 4:  # Check if the mask has an alpha channel
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
    mask = np.expand_dims(mask, 2)
    print(mask.shape, image.shape)

    assert image.shape[:2] == mask.shape[:2]

    # h, w, _ = image.shape
    # grid = 8
    # image = image[:h//grid*grid, :w//grid*grid, :]
    # mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0).astype(np.float32)
    mask = np.expand_dims(mask, 0).astype(np.float32)
    xin = image / 255.0
    mask = mask / 255.0

    print('xin: ',xin.shape)
    print('mask: ',mask.shape)

    G = InpaintGenerator()
    # G.load_weights(args.checkpoint_dir).expect_partial()
    # print(G.history())
    # G.load_weights(args.checkpoint_dir)
    # checkpoint = tf.train.Checkpoint(G)
    # checkpoint.restore(args.checkpoint_dir)
    dummy_input = tf.zeros([1, 512, 512, 3])
    dummy_mask = tf.zeros([1, 512, 512, 1])
    _, _, _ = G(dummy_input,dummy_mask, training=False)
    # for v in G.trainable_variables:
    #     print(v.name, v.shape)
    # with h5py.File('G.weights.h5', 'r') as f:
    #     print(list(f.keys()))
    G.load_weights(args.checkpoint_dir)

    xin = xin*(1.-mask)
    plt.imshow(xin[0])
    plt.show()
    x1, x2, offset_flow = G(xin, mask, training=False)
    batch_predicted = x2
    result = batch_predicted  * mask + xin * (1. - mask)
    cv2.imwrite(args.output, np.array(result[0][:, :, ::-1] * 255))
    # G.save_weights('inpaint_generator_converted.weights.h5')
    G.save_weights('G.h5', save_format='h5')
    # with h5py.File('G.h5', 'r') as f:
    #   print(list(f.keys()))
