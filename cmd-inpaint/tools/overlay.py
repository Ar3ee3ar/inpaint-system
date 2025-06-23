import numpy as np
import cv2
from PIL import Image
import time
# import matplotlib.pyplot as plt
# check corrupted image
# from skimage import io

from . import find_file_path
from .utils import binary_change, up_downscale, gen_tranparent_mask
from .process_pano import Equirectangular,Perspective,inpaint_pano
from .process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from .dataset import Dataset, createAugment
from .py360convert.py360convert import c2e, e2c

# ------------------- overlay function -------------------------

class OverlayGAN():
    def __init__(self, config_data, config_name, type_process_scale = 'resize'):
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
        # ------------------------------------------

        # self.model_type = None
        self.generator = None
        # self.__set_useMask()
        
        # # initial generator model
        # self.model_type = config_data["model_type"]
        # keras.backend.clear_session()
        # if(self.model_type == "p2p"):
        #     self.generator = p2pG.Generator(input_shape=[self.model_input[0],self.model_input[1],3])
        #     self.generator.load_weights(get_file_path(config_data["model_weight"])) # 500
        # elif(self.model_type == "df"):
        #     self.generator = InpaintGenerator()
        #     # G.load_weights(args.checkpoint_dir).expect_partial()
        #     # print(G.history())
        #     # G.load_weights(args.checkpoint_dir)
        #     # checkpoint = tf.train.Checkpoint(G)
        #     # checkpoint.restore(args.checkpoint_dir)
        #     dummy_input = tf.zeros([1, 512, 512, 3])
        #     dummy_mask = tf.zeros([1, 512, 512, 1])
        #     _, _, _ = self.generator(dummy_input,dummy_mask, training=False)
        #     # for v in G.trainable_variables:
        #     #     print(v.name, v.shape)
        #     # with h5py.File('G.weights.h5', 'r') as f:
        #     #     print(list(f.keys()))
        #     self.generator.load_weights(get_file_path(config_data["model_weight"]))
        print('Complete reading config '+config_name)

    # def get_model(self):
    #     print('get_model: ',self.model_type)
    #     print('get_model: ',self.generator)
    #     return (self.generator)

    def set_model_generator(self,generator):
        self.generator = generator
        # print('superclass: ',self.generator)

    def get_haveMask_status(self):
        return self.have_mask

    def set_segMask(self,seg_mask):
        if(seg_mask is not None):
            mask = np.array(seg_mask) # grayscale to ensure image is binary
            self.have_mask = True
        else:
            mask = np.full((self.model_input[0],self.model_input[1],3), 0.0, np.float32)
            self.have_mask = False
        mask_ori = binary_change(mask[:])
        mask_bw = cv2.bitwise_not(mask_ori[:])
        self.in_mask = mask_ori
        # print('mask original')
        # plt.imshow(mask_ori)
        # plt.show()
        # print('mask bitwise')
        # pad_mask = pad_images_to_same_size(mask_ori/255.0) # pad maske (upscale) to match original input size
        pad_mask = up_downscale((mask_ori/255.0).astype(np.float32),(self.img_top_w,self.img_top_h),'upscale',self.type_process_scale) # resize (upscale) mask to original input size
        self.mask_img = pad_mask
        self.mask_img_seamless = (pad_mask[:]*255.0).astype(np.uint8)
        # plt.imshow(cv2.resize(np.array(pad_mask),(512,512)))
        # plt.show()
        # print('pad mask')
        # ----------- change to panorama mask ---------------
        #plt.imshow(cv2.resize(np.array(pad_mask),(512,512)))
        #plt.show()
        equ = Equirectangular((pad_mask[:]*255.0).astype(np.float32), 167, self.coor, -90,'img')    
        mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        self.mask_pano_img = mask_pano_img.astype(np.uint8)
        #plt.imshow(mask_bw)
        #plt.show()
        # --------------- transparent mask for model ---------------
        self.df_mask = gen_tranparent_mask(mask_ori)
        # Convert transparent mask to grayscale
        if self.df_mask.shape[-1] == 4:  # Check if the mask has an alpha channel
            self.df_mask = cv2.cvtColor(self.df_mask, cv2.COLOR_RGBA2GRAY)
        self.df_mask = np.expand_dims(self.df_mask, 2)
        self.df_mask = np.expand_dims(self.df_mask, 0).astype(np.float32)
        self.df_mask = (self.df_mask/255.0).astype(np.float32)
        # return self.have_mask

    def set_useMask(self):
        # print('top')
        # for model
        if(self.mask["top_view"] != ""):
            mask_img = cv2.resize(cv2.imread(find_file_path(self.mask["top_view"])),(self.img_top_h,self.img_top_w)) # for model
            mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
            self.have_mask = True
        else:
            mask_img = np.full((self.img_top_h,self.img_top_w,3), 0.0, np.float32)
            self.have_mask = False
        self.mask_img = mask_img # read from full top mask img
        # mask_img_crop = crop_center(mask_img,(self.model_input[0],self.model_input[1])) # crop (downscale) to match model input size
        mask_img_crop = up_downscale(mask_img,(self.model_input[0],self.model_input[1]),'downscale',self.type_process_scale) # resize (downscale) to match model input size
        mask_img_crop = preprocess_img(mask_img_crop,img_size=(self.model_input[0],self.model_input[1]))
        mask_bw = cv2.bitwise_not(mask_img_crop) # for only u-net
        self.in_mask = mask_bw.astype('float32') # crop to model input
        # for overlay pano
        mask_img_full = mask_img # for model
        mask_img_full = np.where(mask_img_full > 127, 255, 0).astype(np.uint8)
        # seamless mask
        self.mask_img_seamless = mask_img # for model
        # project to equirectangular (get mask image)
        equ = Equirectangular(mask_img_full, 167, self.coor, -90,'img')    
        mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        self.mask_pano_img = mask_pano_img.astype(np.uint8)

        self.df_mask = gen_tranparent_mask(binary_change(mask_img_crop))
        # Convert transparent mask to grayscale
        if self.df_mask.shape[-1] == 4:  # Check if the mask has an alpha channel
            self.df_mask = cv2.cvtColor(self.df_mask, cv2.COLOR_RGBA2GRAY)
        self.df_mask = np.expand_dims(self.df_mask, 2)
        self.df_mask = np.expand_dims(self.df_mask, 0).astype(np.float32)
        self.df_mask = (self.df_mask/255.0).astype(np.float32)
        # print('finish top')


    def seamless(self,source,target,mask):
        # print('seamless function process')
        # mask = self.mask_img_seamless
        # print(type(source),' ',type(target),' ',type(mask))
        # print(source.dtype,' ',target.dtype,' ',mask.dtype)
        # print(source.shape,' ',target.shape,' ',mask.shape)
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

        # Compute center of the bounding box
        center = (x + w // 2, y + h // 2)
        # (h, w) = (self.seamless_center_h, self.seamless_center_w)
        # center = (h,w)
        # center = (1002,901)
        # plt.imshow(np.array(source))
        # plt.show()
        # a = Image.fromarray(source.astype(np.uint8))
        # a.save("D:/KMUTNB/KMUTNB/test_pic/old/Camera/old_output" + '/'+ "img_"+str(0).rjust(6, '0')+"_source.jpg")
        # plt.imshow(np.array(target))
        # plt.show()
        # b = Image.fromarray(((target)).astype(np.uint8))
        # b.save("D:/KMUTNB/KMUTNB/test_pic/old/Camera/old_output" + '/'+ "img_"+str(0).rjust(6, '0')+"_target.jpg")
        # plt.imshow(np.array(mask))
        # plt.show()
        # c = Image.fromarray(((mask)).astype(np.uint8))
        # c.save("D:/KMUTNB/KMUTNB/test_pic/old/Camera/old_output" + '/'+ "img_"+str(0).rjust(6, '0')+"_mask_seam.jpg")
        im_clone = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
        # d = Image.fromarray(((im_clone)).astype(np.uint8))
        # d.save("D:/KMUTNB/KMUTNB/test_pic/old/Camera/old_output" + '/'+ "img_"+str(0).rjust(6, '0')+"_out_seam.jpg")
        # plt.imshow(np.array(im_clone))
        # plt.show()
        return im_clone

    # def save_per(self,img_path,save_path,count):
    #     car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
    #     equ = Perspective(car_pano_img) 
    #     per_img_1000 = equ.GetPerspective(167, self.coor, -90, 1500, 1500)
    #     input_image_im = Image.fromarray((per_img_1000).astype(np.uint8))
    #     input_image_im.save(save_path+count+"_topview_source.jpg")
    

    def equi_to_per(self,img_path,name=''):
        car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))
        equ = Perspective(car_pano_img)
        per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)
        if(name != ''):
            cv2.imwrite("D:/new_car_ds/pano_8k/image_top/"+name+".jpg", per_img_1000)
        return per_img_1000
    
    def save_pano(self,inpaint_fill,save_path,count):
        impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
        impainted_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg")


    def per_to_equi(self,car_path,per_img):
        car_pano_img = preprocess_img(car_path,img_size=(self.img_pano_w,self.img_pano_h))
        start_equ_top = time.time()
        equ = Equirectangular(per_img,167, self.coor, -90)    
        inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        # print("top transform: ",(time.time() - start_equ_top))
        inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
        start_inpaint_top = time.time()
        inpaint_fill = inpaint_pano(inpaint_pano_img,self.mask_pano_img,car_pano_img)
        # print('top inpaint time: ',(time.time() - start_inpaint_top))
        start_save_top = time.time()
        # impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
        # impainted_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg")
        # print('top save time: ',(time.time() - start_save_top))
        #save inpaint img (inpaint mask)
        # inpaint_mask = inpaint_pano(self.mask_pano_img,self.mask_pano_img,car_pano_img)
        # mask_image_im = Image.fromarray(((inpaint_mask)*255.0).astype(np.uint8))
        # mask_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_mask_"+'top'+".jpg")
        # save binary mask panorama
        # bmask_image_im = Image.fromarray(((self.mask_pano_img)).astype(np.uint8))
        # bmask_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_bmask_"+'top'+".jpg")
        # inpaint_fill = (inpaint_fill*255.0).astype(np.uint8)
        # cv2.imwrite(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg", inpaint_fill)
        return inpaint_fill

    def crude_inpaint(self,source, target):
        mask = self.mask_img
        reverse_mask = cv2.bitwise_not(mask)
        source = (source/255.0).astype(np.float32)
        target = (target/255.0).astype(np.float32)
        reverse_mask[reverse_mask != 0] = 1
        result = reverse_mask * target + (1 - reverse_mask) * source
        return result
    
    def overlay(self,main_path,target_coor):
        # try:
        # for i,dir_input in enumerate(dir_list_input):
        all_time = 0.0
        img_path = main_path + "img_"+str(target_coor).rjust(6, '0')+".jpg"
        # print(img_path)
        start_pano = time.time()

        # load image with preprocess
        car_pano_img = preprocess_img(img_path,img_size=(self.img_pano_w,self.img_pano_h))

        # project to perspective (get car image)
        start_proj = time.time()
        equ = Perspective(car_pano_img)    # Load equirectangular image
        per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)  # Specify parameters(FOV, theta, phi, height, width)
        end_proj = time.time()
        # print('top proj: ',end_proj - start_proj)

        # car image
        # per_img = crop_center(per_img_1000,(self.model_input[0],self.model_input[1])) # crop image (downscale) to match model input size
        per_img = up_downscale(per_img_1000,(self.model_input[0], self.model_input[1]),'downscale',self.type_process_scale) # resize (downscale) to match model input size
        car_img_top = preprocess_img(per_img,img_size=(self.model_input[0],self.model_input[1])) # resize to match input model (256,256)

        car_img_top = car_img_top.astype('float32')


        testgen = createAugment(np.array([car_img_top]), np.array([car_img_top]),np.array([self.in_mask]),batch_size=1,dim=(self.model_input[0],self.model_input[1]), shuffle=False, random_mask=False)
        [masked_image, mask], car_img = testgen[0]
        # plt.imshow(masked_image[0])
        # plt.show()

        if(self.model_type == "p2p"):
            # pix2pix ------------------------------------
            car_img = (car_img - 0.5)/0.5 # normalize [-1,1]
            masked_image = (masked_image - 0.5)/0.5 # normalize [-1,1]
            inputs = masked_image # pix2pix
            start_model = time.time()
            predict_image = self.generator(inputs, training=True)
            end_model = time.time()
            # print('model inpaint time: ',(end_model - start_model))
            predict_image_norm = ((predict_image[0]* 0.5) + 0.5).numpy()# de-normalize [0,1]
            # plt.imshow(predict_image_norm)
            # plt.show()
            masked_image = (masked_image* 0.5) + 0.5 # de-normalize [0,1]
            car_img = (car_img* 0.5) + 0.5 # de-normalize [0,1]
            # ------------------------------------------------
        elif(self.model_type == "df"):
            image = np.expand_dims(car_img_top, 0).astype(np.float32)
            # mask = np.expand_dims(mask, 0).astype(np.float32)
            image = image / 255.0
            # plt.imshow(image[0])
            # plt.show()
            masked_image = image*(1. - self.df_mask)
            # plt.imshow(masked_image[0])
            # plt.show()
            # mask = mask / 255.0
            x1, x2, offset_flow = self.generator(masked_image, self.df_mask, training=False)
            batch_predicted = x2
            result = batch_predicted  * self.df_mask + masked_image * (1. - self.df_mask)
            predict_image_norm = np.array(batch_predicted[0])
            # plt.imshow(predict_image_norm)
            # plt.show()

        per_img_1000 = (per_img_1000/255.0).astype(np.float32)

        start_pro_img = time.time()
        predict_image_norm = preprocess_img(predict_image_norm,img_size=(self.model_input[0],self.model_input[1])) # resize to match input model (256,256)
        # predict_image_norm = pad_images_to_same_size(predict_image_norm)
        predict_image_norm = up_downscale(predict_image_norm,(self.img_top_w,self.img_top_h),'upscale',self.type_process_scale) # resize (downscale) to match model input size
        # predict_image_norm = ((1 - mask_img_pano) * per_img_1000) + (mask_img_pano * predict_image_norm)
        predict_image_norm = self.seamless((predict_image_norm*255.0).astype(np.uint8),(per_img_1000*255.0).astype(np.uint8),self.mask_img_seamless)
        # predict_image_norm = predict_image_norm/255.0
        end_pro_img = time.time()
        # print('overlay: ',end_pro_img - start_pro_img)
        final_pano_output = self.per_to_equi(main_path + "img_"+str(target_coor).rjust(6, '0')+".jpg",predict_image_norm)
        return final_pano_output

        # start_reproj = time.time()
        # equ = Equirectangular(predict_image_norm,167, self.coor, -90)
        # inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        # end_reproj = time.time()
        # print('reproj: ',(end_reproj - start_reproj)) 

        # inpaint_fill = inpaint_pano(inpaint_pano_img*255.0,self.mask_pano_img,car_pano_img)
        # # inpaint_mask = inpaint_pano(mask_pano_img,mask_pano_img,car_pano_img)
        # end_pano = time.time()

        # # if(output_mode=="save"):
        # save_path = save_path +  "/img_"+str(target_coor).rjust(6, '0')+"_O.jpg"
        # print(save_path)
        # start_save = time.time()
        # self.save_img(path=save_path,inpaint_img=inpaint_fill)
        # end_save = time.time()
        # print('save_time: ',(end_save - start_save))
        # all_time = all_time + (end_pano - start_pano)
        # print('all_time: ',all_time)

    # def overlay_process(self,main_path,save_path, coor_detail,overwrite):
    #     print(main_path)
    #     target = []
    #     dir_list_input = glob.glob(main_path+'*'+'.jpg')
    #     dir_list_input.sort()
    #     if(not os.path.exists(save_path)):
    #         create_fol(save_path)
    #     else:
    #         if(not bool(strtobool(overwrite))):
    #             for item in dir_list_input[:]: 
    #                 pic_full_name = item.split('/')[-1]
    #                 pic_name = save_path+'/'+pic_full_name[:-4]+'_O.jpg'
    #                 if(os.path.exists(pic_name)):
    #                     dir_list_input.remove(item)
    #     if(len(dir_list_input)>0):
    #         for i in range(len(dir_list_input)):
    #             name_file = (dir_list_input[i].split('/'))[1]
    #             num_file = ((name_file.split('_'))[1])[:-4]
    #             target.append(int(num_file))
        
    #         for coor in tqdm(target):
    #             source_pic = self.find_relate_new(coor_detail,coor,"utm")
    #             self.overlay(main_path,save_path,source_pic,coor,str(coor),coor_detail)

# --------------- inherited for cube overlay -------------------------
class OverlayCube(OverlayGAN):
    def __init__(self,config_data, config_name, generator=None):
        super().__init__(config_data, config_name)
        self.generator = generator
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
        # self.__set_useMask()
        

    # def check_data(self):
    #     print(self.config_data)
    #     print(self.coor)
    #     print('cube class: ',self.generator)
    def set_useMask(self):
        # print('cube')
        for count, side_name in enumerate(self.mask_img):
            if(self.mask[side_name] != ""):
                mask_img = cv2.resize(cv2.imread(find_file_path(self.mask[side_name])),(self.img_top_h,self.img_top_w)) # for model
                mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
                self.have_mask[side_name] = True
            else:
                mask_img = np.full((self.img_top_h,self.img_top_w,3), 0.0, np.float32)
                self.have_mask[side_name] = False
            # for model
            # mask_img = cv2.resize(cv2.imread(find_file_path(self.mask["top_view"])),(self.img_top_h,self.img_top_w)) # for model
            # mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
            self.mask_img[side_name] = mask_img # read from full top mask img
            # mask_img_crop = crop_center(mask_img,(self.model_input[0],self.model_input[1])) # crop (downscale) to match model input size
            mask_img_crop = up_downscale(mask_img,(self.model_input[0],self.model_input[1]),'downscale',self.type_process_scale) # resize (downscale) to match model input size
            mask_img_crop = preprocess_img(mask_img_crop,img_size=(self.model_input[0],self.model_input[1]))
            mask_bw = binary_change(cv2.bitwise_not(mask_img_crop)) # for only u-net
            self.in_mask[side_name] = mask_bw.astype('float32') # crop to model input
            # for overlay pano
            mask_img_full = self.mask_img[side_name] # for model
            mask_img_full = np.where(mask_img_full > 127, 255, 0).astype(np.uint8)
            # seamless mask
            self.mask_img_seamless[side_name] = self.mask_img[side_name] # for model
            # plt.imshow(cv2.resize(self.mask_pano_img,(256,128)))
            # plt.show()
            # --------------- transparent mask for model ---------------
        # for count, side_name in enumerate(self.mask_img):
            self.df_mask[side_name] = gen_tranparent_mask(mask_img_crop)
            # Convert transparent mask to grayscale
            if self.df_mask[side_name].shape[-1] == 4:  # Check if the mask has an alpha channel
                self.df_mask[side_name] = cv2.cvtColor(self.df_mask[side_name], cv2.COLOR_RGBA2GRAY)
            self.df_mask[side_name] = np.expand_dims(self.df_mask[side_name], 2)
            self.df_mask[side_name] = np.expand_dims(self.df_mask[side_name], 0).astype(np.float32)
            self.df_mask[side_name] = (self.df_mask[side_name]/255.0).astype(np.float32)
        # project to equirectangular (get mask image)
        self.mask_pano_img = c2e(list(self.mask_img.values()), h=self.img_pano_h, w=self.img_pano_w, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        # self.mask_pano_img = self.mask_pano_img*255.0
        # plt.imshow(cv2.resize(self.mask_pano_img,(256,128)))
        # plt.show()
        # print('finish cube')
        # return self.have_mask

    def set_segMask(self,seg_mask):
        for count, side_name in enumerate(seg_mask):
            # print(side_name)
            if(seg_mask[side_name] is not None):
                mask = np.array(seg_mask[side_name]) # grayscale to ensure image is binary
                self.have_mask[side_name] = True
            else:
                mask = np.full((self.model_input[0],self.model_input[1],3), 0.0, np.float32)
                self.have_mask[side_name] = False
            mask_ori = binary_change(mask[:])
            mask_bw = cv2.bitwise_not(mask_ori[:])
            self.in_mask[side_name] = mask_ori
            # print('mask original')
            # plt.imshow(mask_ori)
            # plt.show()
            # print('mask bitwise')
            # pad_mask = pad_images_to_same_size(mask_ori/255.0) # pad image (upscale) to match original inpaut size
            pad_mask = up_downscale((mask_ori/255.0).astype(np.float32),(self.img_top_w,self.img_top_h),'upscale',self.type_process_scale) # resize (upscale) to match original input size
            self.mask_img[side_name] = pad_mask
            self.mask_img_seamless[side_name] = (pad_mask[:]*255.0).astype(np.uint8)
            # plt.imshow(cv2.resize(np.array(pad_mask),(512,512)))
            # plt.show()
            # print('pad mask')
        # ----------- change to panorama mask ---------------
        #plt.imshow(cv2.resize(np.array(pad_mask),(512,512)))
        #plt.show()
        # equ = Equirectangular(pad_mask[:]*255.0, 167, self.coor, -90,'img')    
        # mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        # print(len(list(self.mask_img.values())))
        # check_cube_list = list(self.mask_img.values())
        # for i in range(len(check_cube_list)):
        #     plt.imshow(cv2.resize(check_cube_list[i], (512,512)))
        self.mask_pano_img = c2e(list(self.mask_img.values()), h=self.img_pano_h, w=self.img_pano_w, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        self.mask_pano_img = self.mask_pano_img*255.0
        # plt.imshow(cv2.resize(self.mask_pano_img,(256,128)))
        # plt.show()
        # --------------- transparent mask for model ---------------
        for count, side_name in enumerate(seg_mask):
            self.df_mask[side_name] = gen_tranparent_mask(self.in_mask[side_name])
            # Convert transparent mask to grayscale
            if self.df_mask[side_name].shape[-1] == 4:  # Check if the mask has an alpha channel
                self.df_mask[side_name] = cv2.cvtColor(self.df_mask[side_name], cv2.COLOR_RGBA2GRAY)
            self.df_mask[side_name] = np.expand_dims(self.df_mask[side_name], 2)
            self.df_mask[side_name] = np.expand_dims(self.df_mask[side_name], 0).astype(np.float32)
            self.df_mask[side_name] = (self.df_mask[side_name]/255.0).astype(np.float32)

    # def set_useMask(self):
    #     # for model
    #     mask_img = cv2.resize(cv2.imread(get_file_path(self.mask)),(self.img_top_h,self.img_top_w)) # for model
    #     mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
    #     self.mask_img = mask_img # read from full top mask img
    #     mask_img_crop = crop_center(mask_img,(self.model_input[0],self.model_input[1])) # resize to 290,190
    #     mask_img = preprocess_img(mask_img_crop,img_size=(self.model_input[0],self.model_input[1]))
    #     mask_bw = cv2.bitwise_not(mask_img) # for only u-net
    #     self.in_mask = mask_bw.astype('float32') # crop to model input
    #     # for overlay pano
    #     mask_img_full = cv2.resize(cv2.imread(get_file_path(self.config_data["mask_seamless"])),(self.img_top_h,self.img_top_w)) # for model
    #     mask_img_full = np.where(mask_img_full > 127, 255, 0).astype(np.uint8)
    #     # seamless mask
    #     self.mask_img_seamless = cv2.resize(cv2.imread(get_file_path(self.config_data["mask_seamless"])),(self.img_top_h,self.img_top_w)) # for model
    #     # project to equirectangular (get mask image)
    #     equ = Equirectangular(mask_img_full, 167, self.coor, -90,'img')    
    #     mask_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
    #     self.mask_pano_img = mask_pano_img.astype(np.uint8)

    #     self.df_mask = gen_tranparent_mask(binary_change(mask_img_crop))
    #     # Convert transparent mask to grayscale
    #     if self.df_mask.shape[-1] == 4:  # Check if the mask has an alpha channel
    #         self.df_mask = cv2.cvtColor(self.df_mask, cv2.COLOR_RGBA2GRAY)
    #     self.df_mask = np.expand_dims(self.df_mask, 2)
    #     self.df_mask = np.expand_dims(self.df_mask, 0).astype(np.float32)
    #     self.df_mask = self.df_mask/255.0

    def per_to_equi(self,car_path,per_img):
        if isinstance(car_path, str):
            car_pano_img = preprocess_img(car_path,img_size=(self.img_pano_w,self.img_pano_h))
        else:
            car_pano_img = car_path
        inpaint_pano_img = per_img.astype(np.uint8)
        # inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
        # equ = Equirectangular(per_img,167, self.coor, -90)    
        # inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        # inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
        start_inpaint_cube = time.time()
        inpaint_fill = inpaint_pano(inpaint_pano_img,self.mask_pano_img,car_pano_img)
        # print('cube inpaint time: ',(time.time() - start_inpaint_cube))
        start_save_cube = time.time()
        # impainted_image_im = Image.fromarray(((inpaint_fill)*255.0).astype(np.uint8))
        # impainted_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg")
        # print('cube save time: ',(time.time() - start_save_cube))
        #save inpaint img (inpaint mask)
        # inpaint_mask = inpaint_pano(self.mask_pano_img,self.mask_pano_img,car_pano_img)
        # mask_image_im = Image.fromarray(((inpaint_mask)*255.0).astype(np.uint8))
        # mask_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_mask_"+'top'+".jpg")
        # save binary mask panorama
        # bmask_image_im = Image.fromarray(((self.mask_pano_img)).astype(np.uint8))
        # bmask_image_im.save(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_bmask_"+'top'+".jpg")
        # inpaint_fill = (inpaint_fill*255.0).astype(np.uint8)
        # cv2.imwrite(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg", inpaint_fill)
        return inpaint_fill

    def overlay(self,pic_path,target_coor):
        output_cube = []
        # try:
        # for i,dir_input in enumerate(dir_list_input):
        all_time = 0.0
        if isinstance(pic_path, str):
            car_pano_img = pic_path + "img_"+str(target_coor).rjust(6, '0')+".jpg"
            # load image with preprocess
            car_pano_img = preprocess_img(car_pano_img,img_size=(self.img_pano_w,self.img_pano_h))
        else:
            car_pano_img = (np.array(pic_path*255)).astype(np.uint8)
        # print(img_path)
        start_pano = time.time()

        # project to perspective (get car image)
        start_proj = time.time()
        # equ = Perspective(car_pano_img)    # Load equirectangular image
        # per_img_1000 = equ.GetPerspective(167, self.coor, -90, self.img_top_h, self.img_top_w)  # Specify parameters(FOV, theta, phi, height, width)
        image_cube = e2c(car_pano_img, face_w=1000, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom]
        end_proj = time.time()
        # print('cube proj: ',end_proj - start_proj)

        # car image
        for count, side_name in enumerate(self.have_mask):
            # plt.imshow(image_cube[count])
            # plt.show()
            if(self.have_mask[side_name]):
                # per_img = crop_center(image_cube[count],(self.model_input[0],self.model_input[1])) # crop image (downscale) to match model input size 
                per_img = up_downscale(image_cube[count],(self.model_input[0], self.model_input[1]),'downscale',self.type_process_scale) # resize (downscale) to match model input size
                car_img_top = preprocess_img(per_img,img_size=(self.model_input[0],self.model_input[1])) # resize to match input model (256,256)

                car_img_top = car_img_top.astype('float32')

                # print(side_name,': ',np.array([car_img_top]).shape,' ',np.array([self.in_mask[side_name]]).shape)
                testgen = createAugment(np.array([car_img_top]), np.array([car_img_top]),np.array([self.in_mask[side_name]]),batch_size=1,dim=(self.model_input[0],self.model_input[1]), shuffle=False, random_mask=False)
                [masked_image, mask], car_img = testgen[0]
                # plt.imshow(masked_image[0])
                # plt.show()
                start_model = time.time()
                if(self.model_type == "p2p"):
                    # pix2pix ------------------------------------
                    car_img = (car_img - 0.5)/0.5 # normalize [-1,1]
                    masked_image = (masked_image - 0.5)/0.5 # normalize [-1,1]
                    inputs = masked_image # pix2pix
                    # start_model = time.time()
                    predict_image = self.generator(inputs, training=True)
                    # end_model = time.time()
                    # print('model inpaint time: ',(end_model - start_model))
                    predict_image_norm = ((predict_image[0]* 0.5) + 0.5).numpy()# de-normalize [0,1]
                    # plt.imshow(predict_image_norm)
                    # plt.show()
                    masked_image = (masked_image* 0.5) + 0.5 # de-normalize [0,1]
                    car_img = (car_img* 0.5) + 0.5 # de-normalize [0,1]
                    # ------------------------------------------------
                elif(self.model_type == "df"):
                    image = np.expand_dims(car_img_top, 0).astype(np.float32)
                    # mask = np.expand_dims(mask, 0).astype(np.float32)
                    image = image / 255.0
                    # plt.imshow(image[0])
                    # plt.show()
                    masked_image = image*(1. - self.df_mask[side_name])
                    # plt.imshow(masked_image[0])
                    # plt.show()
                    # mask = mask / 255.0
                    x1, x2, offset_flow = self.generator(masked_image, self.df_mask[side_name], training=False)
                    batch_predicted = x2
                    result = batch_predicted  * self.df_mask[side_name] + masked_image * (1. - self.df_mask[side_name])
                    predict_image_norm = np.array(batch_predicted[0])
                    # print('--------------------',side_name)
                    # plt.imshow(predict_image_norm)
                    # plt.show()

                # per_img_1000 = per_img_1000/255.0
                end_model = time.time()
                # print('model inpaint time: ',(end_model - start_model))

                start_pro_img = time.time()
                predict_image_norm = preprocess_img(predict_image_norm,img_size=(self.model_input[0],self.model_input[1])) # resize to match input model (256,256)
                # predict_image_norm = pad_images_to_same_size(predict_image_norm)
                predict_image_norm = up_downscale(predict_image_norm,(self.img_top_w, self.img_top_h),'upscale',self.type_process_scale) # resize image to match real input size
                # predict_image_norm = ((1 - mask_img_pano) * per_img_1000) + (mask_img_pano * predict_image_norm)
                predict_image_norm = self.seamless((predict_image_norm*255.0).astype(np.uint8),(image_cube[count]).astype(np.uint8),self.mask_img_seamless[side_name])
                # predict_image_norm = predict_image_norm/255.0
                end_pro_img = time.time()
                # print('overlay: ',end_pro_img - start_pro_img)
                output_cube.append(predict_image_norm)
            else:
                output_cube.append(self.blank_side)
        start_equ_cube = time.time()
        cube_pano_img = c2e(output_cube, h=self.img_pano_h, w=self.img_pano_w, mode='bilinear', cube_format='list') # [front, right, back, left, top, bottom] # pane inpaint cube to panorama
        # print("cube transform: ",(time.time() - start_equ_cube))
        final_pano_output = self.per_to_equi(car_pano_img,cube_pano_img)

        return final_pano_output

        # start_reproj = time.time()
        # equ = Equirectangular(predict_image_norm,167, self.coor, -90)
        # inpaint_pano_img,_ = equ.GetEquirec(self.img_pano_h,self.img_pano_w)
        # end_reproj = time.time()
        # print('reproj: ',(end_reproj - start_reproj)) 

        # inpaint_fill = inpaint_pano(inpaint_pano_img*255.0,self.mask_pano_img,car_pano_img)
        # # inpaint_mask = inpaint_pano(mask_pano_img,mask_pano_img,car_pano_img)
        # end_pano = time.time()

        # # if(output_mode=="save"):
        # save_path = save_path +  "/img_"+str(target_coor).rjust(6, '0')+"_O.jpg"
        # print(save_path)
        # start_save = time.time()
        # self.save_img(path=save_path,inpaint_img=inpaint_fill)
        # end_save = time.time()
        # print('save_time: ',(end_save - start_save))
        # all_time = all_time + (end_pano - start_pano)
        # print('all_time: ',all_time)