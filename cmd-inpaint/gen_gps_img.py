import argparse
import cv2
import numpy as np
import os
import time
from PIL import Image
import time
import utm
import scipy.sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import glob

# sys.path.append("..")

from tools.process_img import preprocess_img, crop_center, pad_images_to_same_size, resize
from tools.process_pano import Equirectangular,Perspective,inpaint_pano
from initial_gps import *

# degree สะพาน (167, 0, -90, 1000, 1000)
# degree สะพาน (167, 27, -80, 1000, 1000)

def laplacian_matrix(n, m):
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


def poisson_edit(source, target, mask, offset):
    """The poisson blending function. 

    Refer to: 
    Perez et. al., "Poisson Image Editing", 2003.
    """

    # Assume: 
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min
        
    M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
    source = cv2.warpAffine(source,M,(x_range,y_range))
    # cv2.imshow("source_in_poison",source)
        
    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1
    # cv2.imshow("mask_in_poison",cv2.resize(mask,(512,512)))
    #mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    mat_A = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity    
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    # corners
    # mask[0, 0]
    # mask[0, y_range-1]
    # mask[x_range-1, 0]
    # mask[x_range-1, y_range-1]

    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()    
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()        

        #concat = source_flat*mask_flat + target_flat*(1-mask_flat)
        
        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b)
        #print(x.shape)
        x = x.reshape((y_range, x_range))
        #print(x.shape)
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        #x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #print(x.shape)

        target[y_min:y_max, x_min:x_max, channel] = x

    # cv2.waitKey() 
    # cv2.destroyAllWindows()

    return target

def seamless(source,target):
    mask = mask_img
    # mask = cv2.imread(mask_path)
    # mask = cv2.resize(mask,(2000,2000))
    # mask[mask != 0] = 255
    center = (1002,901)
    # center = (1000,910)
    # print(center)
    im_clone = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)

    # cv2.imshow("img",cv2.resize(mask,(256,256)))
    # cv2.waitKey() 
    # cv2.destroyAllWindows()
    return im_clone

def convert_yaw_to_360(yaw):
    # Ensure yaw is in the range 0 to 360 degrees
    yaw_360 = (yaw + 360) % 360
    return yaw_360

def find_relate(coor_detail,target_pic,type_coor):
    # (np.abs(coor_detail["y"][target_pic] - coor_detail["y"][i]) >= 7)
    qt = [-12,12]
    theta = coor_detail["yaw"][target_pic]
    # print([coor_detail["lat"][target_pic], coor_detail["lon"][target_pic]])
    if(type_coor == "latlon"):
        target_coor = latlon_to_utm([coor_detail["x"][target_pic], coor_detail["y"][target_pic]])
    else:
        target_coor = [coor_detail["x"][target_pic], coor_detail["y"][target_pic]]
    xt, yt = target_coor[0:2]
    # print(xt,' ',yt)
    range_xq = []
    range_yq = []
    for range_qt in qt:
        xq = range_qt*np.sin(np.radians(theta + 180)) + xt
        yq = range_qt*np.cos(np.radians(theta + 180)) + yt
        range_xq.append(xq)
        range_yq.append(yq)
    range_relate_coor = []
    for range_coor in range(len(range_xq)):
        if(type_coor == "latlon"):
            relate_coor = utm_to_latlon((range_xq[range_coor],range_yq[range_coor],target_coor[2],target_coor[3]))
        else:
            relate_coor = [range_xq[range_coor],range_yq[range_coor]]
        range_relate_coor.append(relate_coor)
    # print('relate coor latlon: ',range_relate_coor[0],' , ',range_relate_coor[1])
    min_x_dist = 9999999
    min_y_dist = 9999999
    sum_dist = 9999999
    min_pic = target_pic
    for i in range(0,len(coor_detail["x"])):
        left_x_dist = np.abs(range_relate_coor[0][0] - coor_detail["x"][i])
        right_x_dist = np.abs(range_relate_coor[1][0] - coor_detail["x"][i])
        left_y_dist = np.abs(range_relate_coor[0][1] - coor_detail["y"][i])
        right_y_dist = np.abs(range_relate_coor[1][1] - coor_detail["y"][i])
        # print(i,': ',left_x_dist,' , ',left_y_dist)
        # print(i,': ',right_x_dist,' , ',right_y_dist,'/n')
        if((left_x_dist + left_y_dist < sum_dist) or ((right_x_dist + right_y_dist < sum_dist))):
            sum_dist = min_x_dist + min_y_dist
            if(((left_x_dist< min_x_dist) or (left_y_dist< min_y_dist)) or ((right_x_dist < min_x_dist) or (right_y_dist < min_y_dist))):
                # print(coor_detail["y"][target_pic] - range_relate_coor[0][1])
                # print(coor_detail["y"][target_pic] - range_relate_coor[1][1])
                if(((left_x_dist< min_x_dist) or (left_y_dist< min_y_dist)) and (np.abs(coor_detail["y"][target_pic] - coor_detail["y"][i]) >= 7)):
                    min_x_dist = left_x_dist
                    min_y_dist = left_y_dist
                    min_pic = i
                    sum_dist = min_x_dist + min_y_dist
                if(((right_x_dist< min_x_dist) or (right_y_dist< min_y_dist)) and (np.abs(coor_detail["y"][target_pic] - coor_detail["y"][i]) >= 7)):
                    min_x_dist = right_x_dist
                    min_y_dist = right_y_dist
                    min_pic = i
                    sum_dist = min_x_dist + min_y_dist
            # print('min: ',i,' ',min_x_dist,' , ',min_y_dist)
        # elif(((left_x_dist> min_x_dist) and (left_y_dist> min_y_dist)) or ((right_x_dist > min_x_dist) and (right_y_dist > min_y_dist))):
        #     break
    # print(target_pic,' :use=> ',min_pic)
    return min_pic

def save_per(img_path,save_path,count):
    car_pano_img = preprocess_img(img_path,img_size=(8192,4096))
    equ = Perspective(car_pano_img)    # Load equirectangular image
    per_img_1000 = equ.GetPerspective(167, 27, -90, 1500, 1500)  # Specify parameters(FOV, theta, phi, height, width)
    input_image_im = Image.fromarray((per_img_1000).astype(np.uint8))
    input_image_im.save(save_path+count+"_topview_source.jpg")


def equi_to_per(img_path,name=''):
    car_pano_img = preprocess_img(img_path,img_size=(8192,4096))
    equ = Perspective(car_pano_img)    # Load equirectangular image
    per_img_1000 = equ.GetPerspective(167, 27, -90, 2000, 2000)  # Specify parameters(FOV, theta, phi, height, width)
    if(name != ''):
        cv2.imwrite("D:/new_car_ds/pano_8k/image_top/"+name+".jpg", per_img_1000)
    # per_img = crop_center(per_img_1000,(256,256))
    # save input image
    # save_path = "D:/new_car_ds/pano_8k/pano_8k_per/"
    # input_image_im = Image.fromarray((per_img_1000).astype(np.uint8))
    # input_image_im.save("D:/new_car_ds/pano_8k/IMAGE_OUTPUT_per_1500/"+count+"_topview_source.jpg")

    # per_image_im = Image.fromarray((per_img).astype(np.uint8))
    # per_image_im.save("D:/new_car_ds/pano_8k/IMAGE_OUTPUT_crop/"+count+"_topview_source.jpg")
    return per_img_1000

def per_to_equi(save_path,car_path,per_img,count):
    # mask_path = "car_ds/pic/mask/8k_focus_simple.jpg"
    car_pano_img = cv2.imread(car_path)
    # per_img = preprocess_img(per_path,img_size=(1000,1000))
    # mask_img_pano = cv2.imread(mask_path)
    # mask_img_pano = cv2.resize(mask_img_pano,(2000,2000))
    # # mask_img_pano = preprocess_img(mask_path,img_size=(1000,1000)) # for inpaint in pano
    # # mask_img_crop = crop_center(mask_img_pano,(256,256))
    # equ = Equirectangular(mask_img_pano,167, 27, -90,'img')    
    # mask_pano_img,mask = equ.GetEquirec(4096,8192)
    # mask_pano_img = mask_pano_img.astype(np.uint8)

    equ = Equirectangular(per_img,167, 27, -90)    
    inpaint_pano_img,_ = equ.GetEquirec(4096,8192)
    inpaint_pano_img = inpaint_pano_img.astype(np.uint8)
    inpaint_fill = inpaint_pano(inpaint_pano_img,mask_pano_img,car_pano_img)
    inpaint_fill = (inpaint_fill*255.0).astype(np.uint8)
    # print(inpaint_fill[0])

    # cv2.imshow("car_pano_img", cv2.resize(car_pano_img,(512,256)))
    # cv2.imshow("inpaint_pano_img", cv2.resize(inpaint_pano_img,(512,256)))
    # cv2.imshow("mask_pano_img", cv2.resize(mask_pano_img,(512,256)))
    # cv2.imshow("inpaint_fill", inpaint_fill.astype(np.uint8))

    # save inpaint_img
    # save_path = "D:/new_car_ds/pano_8k/pano_poison/"
    # input_image_im = Image.fromarray((inpaint_fill).astype(np.uint8))
    # input_image_im.save(save_path+count+"_test_inpaint.jpg")
    cv2.imwrite(save_path + '/'+ "img_"+str(count).rjust(6, '0')+"_O.jpg", inpaint_fill)

    # #save per img
    # per_image_im = Image.fromarray((inpaint_pano_img).astype(np.uint8))
    # per_image_im.save(save_path+count+"_inpaint.jpg")
    # cv2.imwrite("D:/new_car_ds/pano_8k/image_top/"+count+"_afterfill.jpg", per_img)
    # cv2.imwrite(save_path + 'Top/'+ count+"_afterfill.jpg", per_img)

    # #save original img
    # ori_image_im = Image.fromarray((car_pano_img).astype(np.uint8))
    # ori_image_im.save(save_path+count+"_test_inpaint_1ori.jpg")

    # #save mask (projecting) img
    # mask_image_im = Image.fromarray((mask_pano_img).astype(np.uint8))
    # mask_image_im.save(save_path+count+"_test_reproject_mask.jpg")

    # #save mask (crop) img
    # mask_image_crop = Image.fromarray((mask_img_crop).astype(np.uint8))
    # mask_image_crop.save(save_path+count+"_test_mask_crop.jpg")

    # cv2.imshow("car_pano_img", cv2.resize(car_pano_img,(256,512)))
    # cv2.imshow("per_img", cv2.resize(per_img,(256,256)))
    # # cv2.imshow("inpaint_pano_img",cv2.resize(inpaint_pano_img,(256,512)))
    # cv2.imshow("inpaint_fill", cv2.resize(inpaint_fill,(256,512)))
    # cv2.waitKey() 
    # cv2.destroyAllWindows()

def latlon_to_utm(coor):
    utm_coor = utm.from_latlon(coor[0],coor[1])
    # print(utm_coor)
    return utm_coor

def utm_to_latlon(coor):
    # print(coor)
    latlon_coor = utm.to_latlon(coor[0],coor[1],coor[2],coor[3])
    return latlon_coor

def rotate(image,degree):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), -degree, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def crude_inpaint(source, target, mask_path):
    mask = cv2.imread(mask_path)
    mask = cv2.bitwise_not(mask)
    source = source/255.0
    target = target/255.0
    mask[mask != 0] = 1
    result = mask * target + (1 - mask) * source
    return result

def overlay(main_path,save_path,source_path, target_path,count,coor_detail):
    mask_path = "car_ds/pic/mask/8k_focus_simple.jpg"
    # mask = preprocess_img(mask_path,img_size=(1000,1000))
    # mask = cv2.bitwise_not(mask)
    # relate_image = move(relate_image)
    # target_image = target_image/255.0
    # relate_image = relate_image/255.0
    # mask = mask/255.0
    # fill_image = mask * target_image + (1 - mask) * relate_image
    # fill_image = cv2.cvtColor((fill_image*255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
    offset = (0,0)
    source = cv2.imread(main_path + "img_"+str(source_path).rjust(6, '0')+".jpg") 
    target = cv2.imread(main_path + "img_"+str(target_path).rjust(6, '0')+".jpg") 
    source = equi_to_per(source)
    target = equi_to_per(target)
    dist_utm_x = coor_detail["x"][source_path] - coor_detail["x"][target_path]
    dist_utm_y = coor_detail["y"][source_path] - coor_detail["y"][target_path]
    dist_degree = convert_yaw_to_360(coor_detail["yaw"][source_path]) - convert_yaw_to_360(coor_detail["yaw"][target_path])
    # print('dist_x : ',coor_detail["x"][source_path] - coor_detail["x"][target_path])
    # print('dist_y : ',coor_detail["y"][source_path] - coor_detail["y"][target_path])
    # print('dist_yaw : ',convert_yaw_to_360(coor_detail["yaw"][source_path]) - convert_yaw_to_360(coor_detail["yaw"][target_path]))
    new_coor =  find_new_coor(dist_utm_x,dist_utm_y,dist_degree)
    source = cv2.warpAffine(source, np.float32([[1, 0, m_to_px(new_coor[0]-dist_utm_x)], [0, 1, m_to_px(new_coor[1]-dist_utm_y)]]), (2000,2000))
    source = rotate(source,convert_yaw_to_360(coor_detail["yaw"][source_path]) - convert_yaw_to_360(coor_detail["yaw"][target_path]))
    # print('new_coor : ',new_coor[0],' , ',new_coor[1])
    # print("move_coor : ( ",dist_utm_x - new_coor[0]," , ",dist_utm_y - new_coor[1]," )")
    # print("shift: ( ",m_to_px(new_coor[0]-dist_utm_x),' , ',m_to_px(new_coor[1]-dist_utm_y),' )')
    if(source_path > target_path):
        # print('pic forward move down')
        source = move(source,"down",dist_utm_y)
    elif(source_path < target_path):
        # print("pic backward move up")
        source = move(source,"up",dist_utm_y)
    else:
        # print("same pic move down")
        source = move(source,"down",7)
    # source = rotate(source,convert_yaw_to_360(coor_detail["yaw"][source_path]) - convert_yaw_to_360(coor_detail["yaw"][target_path]))
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # mask = cv2.resize(mask,(2000,2000))
    # cv2.imshow("source",cv2.resize(source, (512, 512)))
    # cv2.imshow("target",cv2.resize(target, (512, 512)))
    # cv2.imshow("mask",cv2.resize(mask, (512, 512)))
    # print('poison')
    # result = poisson_edit(source,target, mask/255.0, offset)
    # result = crude_inpaint(source,target,mask_path)
    result = seamless(source,target)
    # print('projection')
    # save_path = main_path
    per_to_equi(save_path, main_path + "img_"+str(target_path).rjust(6, '0')+".jpg",result,count)
    # resize_target = cv2.resize(result, (512, 512))
    # cv2.imshow("img",resize_target)
    # cv2.waitKey() 
    # cv2.destroyAllWindows()
    # pass

# def blend(image):
#     pass

def move(img,direction,dist):
    if(direction == "up"):
        T = np.float32([[1, 0, 0], [0, 1, 495]]) # [1,0,x],[0,1,y]
    elif(direction == "down"):
        T = np.float32([[1, 0, 0], [0, 1, -495]]) # [1,0,x],[0,1,y]
    # We use warpAffine to transform 
    # the image using the matrix, T 
    img_translation = cv2.warpAffine(img, T, (2000, 2000))
    return img_translation

# def move_mask():
#     mask_path = "D:/inpaint_gan/car_ds/pic/mask/rect_test2_mask.jpg"
#     mask_img_pano = preprocess_img(mask_path,img_size=(1000,1000)) # for inpaint in pano
#     T = np.float32([[1, 0, 0], [0, 1, -50]]) # [1,0,x],[0,1,y]
#     # We use warpAffine to transform 
#     # the image using the matrix, T 
#     img_translation = cv2.warpAffine(mask_img_pano, T, (1000, 1000))

#     #save mask
#     save_path = "D:/new_car_ds/pano_8k/IMAGE_OUTPUT_per/"
#     input_image_im = Image.fromarray((img_translation).astype(np.uint8))
#     input_image_im.save(save_path+"rect_test2_mask.jpg")

# def resize_pano(car_path,want_size,count):
#     car_pano_img = preprocess_img(car_path,img_size=want_size)
#     #save mask
#     save_path = "D:/new_car_ds/pano_8k/IMAGE_OUTPUT_per/"
#     input_image_im = Image.fromarray((car_pano_img).astype(np.uint8))
#     input_image_im.save(save_path+str(want_size)+"_"+str(count)+"rect_test2_mask.jpg")

def find_new_coor(x,y,degree):
    x_new = x*np.cos(np.radians(degree)) + y*np.sin(np.radians(degree))
    y_new = -x*np.sin(np.radians(degree)) + y*np.cos(np.radians(degree))
    return (x_new,y_new)

def m_to_px(X):
    Y = ((29.663*X) +2.644 - 1)
    return Y 

# def move_test(source_path,target_path,source_coor,target_coor):
#     source = cv2.resize(cv2.imread("D:/new_car_ds/C1/C1_per_1500/"+str(source_path)+"_topview_source.jpg"),(2000,2000))
#     target = cv2.resize(cv2.imread("D:/new_car_ds/C1/C1_per_1500/"+str(target_path)+"_topview_source.jpg"),(2000,2000))
#     height, width = source.shape[:2] 
#     print(height,' ',width)
#     # quarter_height, quarter_width = height / 4, width / 4
#     dist_utm_x = target_coor[0] - source_coor[0]
#     dist_utm_y = target_coor[1] - source_coor[1]
#     dist_degree = convert_yaw_to_360(coor_detail["yaw"][source_path]) - convert_yaw_to_360(coor_detail["yaw"][target_path])
#     print('before: ',coor_detail["yaw"][source_path],' , ',coor_detail["yaw"][target_path])
#     print('after: ',convert_yaw_to_360(coor_detail["yaw"][source_path]),' , ',convert_yaw_to_360(coor_detail["yaw"][target_path]))
#     print(dist_utm_x, dist_utm_y, dist_degree)
#     dist_px_x = 0
#     dist_px_y = 70.114 * dist_utm_y
#     print(dist_px_x,dist_px_y)
#     T = np.float32([[1, 0, dist_px_x], [0, 1, dist_px_y]]) # [1,0,x],[0,1,y]
#     # We use warpAffine to transform 
#     # the image using the matrix, T 
#     # img_translation = rotate(source,dist_degree)
#     # img_translation = cv2.warpAffine(img_translation, T, (width, height))
#     new_coor =  find_new_coor(dist_utm_x,dist_utm_y,dist_degree)
#     print("old_coor : ( ",dist_utm_x," , ", dist_utm_y," )")
#     print("new_coor : ( ",new_coor[0]," , ", new_coor[1]," )")
#     print("move_coor : ( ", new_coor[0]-dist_utm_x," , ", new_coor[1]-dist_utm_y," )")
#     print("shift: ( ",m_to_px(dist_utm_x-new_coor[0]),' , ',m_to_px(dist_utm_y-new_coor[1]),' )')
#     # img_translation = cv2.warpAffine(source, T, (2000, 2000))
#     # img_translation = rotate(img_translation,dist_degree)
#     # img_translation = cv2.warpAffine(img_translation, np.float32([[1, 0, 33], [0, 1, 30]]), (width, height))

#     img_translation = cv2.warpAffine(source, np.float32([[1, 0, m_to_px(dist_utm_x-new_coor[0])], [0, 1, m_to_px(dist_utm_y-new_coor[1])]]), (2000,2000))
#     img_translation = rotate(img_translation,dist_degree)
#     img_translation = cv2.warpAffine(img_translation, T, (width, height))

#     resize_target = cv2.resize(target, (512, 512))
#     resize_img_translation = cv2.resize(img_translation, (512, 512))
#     added_image = cv2.addWeighted(resize_target,0.4,resize_img_translation,0.4,0)

#     cv2.imshow("Originalimage", resize_target) 
#     cv2.imshow('Translation', resize_img_translation)
#     cv2.imshow("img",added_image)
#     cv2.waitKey() 
#     cv2.destroyAllWindows()

def create_fol(path):
    if not os.path.exists(path): 
        os.makedirs(path)
    else:
        print('already create '+path)

def overlay_process(main_path,save_path, coor_detail):
    print(main_path)
    target = []
    dir_list_input = glob.glob(main_path+'*'+'.jpg')
    dir_list_input.sort()
    # source = []
    if(not os.path.exists(save_path)):
        create_fol(save_path)
    else:
        # create_fol(main_path+'Top')
        for item in dir_list_input[:]:  # Use a copy of the list
            pic_full_name = item.split('\\')[-1]
            pic_name = save_path+'/'+pic_full_name[:-4]+'_O.jpg'
            # print(pic_name)
            # break
            if(os.path.exists(pic_name)):
                dir_list_input.remove(item)
        # print(dir_list_input[0:3])

    # print(dir_list_input[0])
    if(len(dir_list_input)>0):
        for i in range(len(dir_list_input)):
            name_file = (dir_list_input[i].split('\\'))[1]
            num_file = ((name_file.split('_'))[1])[:-4]
            # print(num_file)
            target.append(int(num_file))
            # break

        for coor in tqdm(target):
            # print('-------',coor,'-----------')
            source_pic = find_relate(coor_detail,coor,"utm")
            overlay(main_path,save_path,source_pic,coor,str(coor),coor_detail)
            # break

# def save_per_process():
#     for coor in tqdm(range(2061)):
#         save_per(img_path="D:/new_car_ds/C1/camera/img_"+str(coor).rjust(6, '0')+".jpg",save_path="D:/new_car_ds/C1/C1_per_1500/",count=str(coor))

# def find_equ_move():
#     # source_pic = 12
#     target_pic = 2074
#     # source_path = "D:/new_car_ds/C1/C1_per_1500/"+str(source_pic)+"_topview_source.jpg"
#     # target_path = "D:/new_car_ds/C1/C1_per_1500/"+str(target_pic)+"_topview_source.jpg"
#     source_pic = find_relate(coor_detail,target_pic,"utm")
#     source_coor = (coor_detail["x"][source_pic], coor_detail["y"][source_pic])
#     target_coor = (coor_detail["x"][target_pic], coor_detail["y"][target_pic])
#     move_test(source_pic,target_pic,source_coor,target_coor)

def find_fol(main_input):
    dir_list_input = glob.glob(main_input+'*/*/'+'Camera/')
    for i in range(len(dir_list_input)):
        dir_list_input[i] = dir_list_input[i].replace('\\','/')
    # print(dir_list_input)
    return dir_list_input

def main():
    main_input = "E:/ProcessMMS/" #NOTE: change man path here
    path_input = find_fol(main_input)
    for i in range(len(path_input)):
        line_name = (path_input[i].split('/'))[3]
        save_path = path_input[i]+line_name+'_output'
        # print(save_path)
        dir_list_input = glob.glob(path_input[i]+'*'+'_camera_position.txt')
        if(len(dir_list_input)>0):
            # print(dir_list_input)
            coor_path = dir_list_input[0]
            coor_detail = {
                "x":[],
                "y": [],
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
                        coor_detail["roll"].append(float(coor_list[7]))
                        coor_detail["pitch"].append(float(coor_list[8]))
                        coor_detail["yaw"].append(float(coor_list[6]))
                        # print(count)
            # print(coor_detail["x"][0])
            overlay_process(path_input[i],save_path,coor_detail)

if __name__ == '__main__':
    main()