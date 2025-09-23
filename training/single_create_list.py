import argparse
import glob
import cv2
import os
from sklearn.model_selection import train_test_split

main_dir = ""
# 'car_ds/pic/mask/8k_focus_shape.jpg'

# Description: command ไว้ทำตามคำสั่ง
def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str, default='', help="create to create per|cube dataset")
    parser.add_argument("--config_name", "-config_name",type=str, default='', help="folder name to store config dataset")
    parser.add_argument("--fol_img", "-fol_img",type=str, default='', help="folder that store dataset")
    parser.add_argument("--mask", "-mask",type=str, default='', help="img mask path")
    parser.add_argument("--artifact_path", "-artifact_path",type=str, default='', help="artifact path")
    parser.add_argument("--patt_name", "-patt_name",type=str, default='.jpg', help="select pattern file name")
    arg = parser.parse_args()
    return arg

def get_mask_file(ds_path, path_list):
    for i in range(len(path_list)):
        path_list[i] = path_list[i].replace('.jpg', '_mask.png')
    return path_list


def get_file_name(ds_path,path_list):
    # print(len(path_list))
    # print(ds_path)
    # print(path_list[0])
    for i in range(len(path_list)):
        # print(path_list[i])
        # input img process
        split_name_input = path_list[i].split('\\') # split path name
        path_list[i] = ds_path+split_name_input[1]
    return path_list

def create_fol(path):
    if not os.path.exists(path): 
        os.makedirs(path+'/test')
        os.makedirs(path+'/train')
        os.makedirs(path+'/val')
    else:
        print('already create '+path)

def cat_cube_face(cube_list, face):
    name_cube_list = []
    for i in range(len(cube_list)):
        name_pic = (cube_list[i].split('/'))[-1]
        # print('name_pic: ',name_pic,' <- ',cube_list[i])
        cube_face = (name_pic.split('_'))[-1].split('.')
        # print('cube_face: ',cube_face)
        if(cube_face[0] == face):
           name_cube_list.append(cube_list[i])
    
    return name_cube_list

def remove_artifact(input,output,mask=None,pattern_name='',artifact_list=[]):
    # print(artifact_list[0]+pattern_name,' == ',(input[0].split('/'))[3])
    for i in range(len(input)):
        for j in range(len(artifact_list)):
            try:
                if(artifact_list[j]+pattern_name == (input[i].split('/'))[3]):
                    input[i] = 0
                    output[i] =0
                    if mask is not None:
                        mask[i] = 0
            except:
                pass

    input = [x for x in input if x != 0]
    output = [x for x in output if x != 0]
    if mask is not None:
        mask = [x for x in mask if x != 0]

    if (mask is not None):
        return input,output,mask
    elif (mask is None):
        return input,output

def perspective_ds(fol_pic_name,mask_path,data_file_name):
    path_input_list = []
    path_output_list = []
    mask_list = []
    artifact_list = []
    
    if(_argparse().artifact_path != ''):
        with open("car_ds/train_test_config/"+_argparse().artifact_path) as f:
            line = f.readlines()
            for num_pic in line:
                artifact_list.append(str(num_pic.strip()))

    create_fol(main_dir+'car_ds/train_test_config/'+data_file_name)
    # 1000 x 1000
    path_input = main_dir + "car_ds/pic/"+fol_pic_name+"/"
    dir_list_input = glob.glob(path_input+'*'+_argparse().patt_name)
    # print(dir_list_input)
    dir_list_input.sort()
    # path_output = main_dir + "car_ds/pic/image_test2/"
    # dir_list_output = glob.glob(path_output+"*_afterfill.jpg")
    dir_list_output = dir_list_input[:]
    dir_list_mask = dir_list_input[:]
    if(mask_path[:-4] == ".jpg" or mask_path[:-4] == ".png"):
        mask_img = "car_ds/pic/"+mask_path # one mask
        mask_list = mask_list + ([mask_img] * len(dir_list_input))
    else:
        # path_mask = main_dir + "car_ds/pic/"+fol_pic_name+"/"
        # dir_list_mask = glob.glob(path_mask+'*'+'mask.png')
        # # print(len(dir_list_mask))
        # dir_list_mask.sort()
        # dir_list_mask = get_file_name("car_ds/pic/"+fol_pic_name+"/", dir_list_mask)
        # mask_list = mask_list + dir_list_mask
        dir_list_mask = get_mask_file("car_ds/pic/"+fol_pic_name+"/",dir_list_mask)
        dir_list_mask = get_file_name("car_ds/pic/"+fol_pic_name+"/", dir_list_mask)
        mask_list = mask_list + dir_list_mask

    # mask_img = "car_ds/pic/"+mask_path # one mask
    # get only file_name
    dir_list_input = get_file_name("car_ds/pic/"+fol_pic_name+"/",dir_list_input)
    dir_list_output = get_file_name("car_ds/pic/"+fol_pic_name+"/",dir_list_output)
    # mask_img = get_file_name("car_ds/mask/",[mask_img])
    # append to list
    path_input_list = path_input_list + dir_list_input
    path_output_list = path_output_list + dir_list_output
    # mask_list = mask_list + ([mask_img] * len(dir_list_input))
    print(dir_list_input[0])
    print(dir_list_output[0])
    print(dir_list_mask[0])
    if(_argparse().artifact_path != ''):
        path_input_list,path_output_list,mask_list = remove_artifact(path_input_list,path_output_list,mask_list,pattern_name='_topview_target.jpg',artifact_list = artifact_list)
    x_train, x_test, y_train, y_test,mask_train,mask_test = train_test_split(path_input_list, path_output_list,mask_list, train_size=0.70, random_state=123)
    x_test, x_val, y_test, y_val,mask_test,mask_val = train_test_split(x_test, y_test,mask_test, train_size=0.67, random_state=123)
    print(len(x_train))

    main_folder_path = 'car_ds/train_test_config/'+data_file_name+'/'
    # train data
    with open(main_dir + main_folder_path + 'train/input.txt', 'w') as f:
        f.write('\n'.join(x_train))
    with open(main_dir + main_folder_path+'train/masks.txt', 'w') as f:
        f.write('\n'.join(mask_train))
    with open(main_dir + main_folder_path+'train/output.txt', 'w') as f:
        f.write('\n'.join(y_train))

    # test data
    with open(main_dir + main_folder_path+'test/input.txt', 'w') as f:
        f.write('\n'.join(x_test))
    with open(main_dir + main_folder_path+'test/masks.txt', 'w') as f:
        f.write('\n'.join(mask_test))
    with open(main_dir + main_folder_path+'test/output.txt', 'w') as f:
        f.write('\n'.join(y_test))

    # val data
    with open(main_dir + main_folder_path+'val/input.txt', 'w') as f:
        f.write('\n'.join(x_val))
    with open(main_dir + main_folder_path+'val/masks.txt', 'w') as f:
        f.write('\n'.join(mask_val))
    with open(main_dir + main_folder_path+'val/output.txt', 'w') as f:
        f.write('\n'.join(y_val))



def cube_ds(fol_pic_name,mask_path,data_file_name):
    # 'LB_0_000001_B.jpg'
    # artifact_per_list = []
    artifact_cube_list = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_val = []
    y_val = []
    mask_train = []
    mask_test = []
    mask_val = []

    face_list = ['F', 'B', 'L', 'R']
    for face in face_list:
        with open(main_dir+'car_ds/train_test_config/artifact_cube/'+face+'.txt') as f:
            line = f.readlines()
            for num_pic in line:
                name_pic = "LB_0_"+str(num_pic.strip()).rjust(6, '0')+"_"+face+".jpg"
                artifact_cube_list.append(name_pic)
        # 1000 x 1000 + crop (256)
        path_input_list = []
        path_output_list = []
        mask_list = []
        config_list = []
        path_input = main_dir + "car_ds/pic/"+fol_pic_name+"/"
        dir_list_input = glob.glob(path_input+"LB*.jpg")

        dir_list_input.sort()
        print(len(dir_list_input))
        # dir_list_output = dir_list_input[:] # clone input list
        # get only file_name
        dir_list_input = get_file_name("car_ds/pic/"+fol_pic_name+"/",dir_list_input)
        path_input_list = cat_cube_face(dir_list_input,face)
        path_output_list = path_input_list[:]
        mask_img = 'car_ds/pic/mask/car_equip_mask_'+face+'.jpg'
        # remove artifact
        print('data ',face,': ',len(path_input_list))
        path_input_list,path_output_list = remove_artifact(path_input_list,path_output_list,mask=None,pattern_name='',artifact_list = artifact_cube_list)
        # report result
        print('remove artifact data ',face,': ',len(path_input_list))
        # train test split
        mask_list = [mask_img]*len(path_input_list)
        x_train_all_per, x_test_per, y_train_all_per, y_test_per, mask_train_all_per, mask_test_per = train_test_split(path_input_list, path_output_list,mask_list, train_size=1100, random_state=123)
        x_train_per, x_val_per, y_train_per, y_val_per, mask_train_per, mask_val_per = train_test_split(x_train_all_per, y_train_all_per,mask_train_all_per, train_size=1000, random_state=123)
        # append to list
        x_train = x_train + x_train_per
        y_train = y_train + y_train_per
        x_test = x_test + x_test_per
        y_test = y_test + y_test_per
        x_val = x_val + x_val_per
        y_val = y_val + y_val_per
        mask_train = mask_train + mask_train_per
        mask_test = mask_test + mask_test_per
        mask_val = mask_val + mask_val_per


    # report result
    print('data per: ',len(path_input_list))
    print('train data: ',len(x_train))
    print('test data: ',len(x_test))
    print('val data: ',len(x_val))

    main_folder_path = 'car_ds/train_test_config/'+data_file_name+'/'
    # train data
    with open(main_dir + main_folder_path + 'train/input.txt', 'w') as f:
        f.write('\n'.join(x_train))
    with open(main_dir + main_folder_path+'train/output.txt', 'w') as f:
        f.write('\n'.join(y_train))
    with open(main_dir + main_folder_path+'train/masks.txt', 'w') as f:
        f.write('\n'.join(mask_train))
    # test data
    with open(main_dir + main_folder_path + 'test/input.txt', 'w') as f:
        f.write('\n'.join(x_test))
    with open(main_dir + main_folder_path+'test/output.txt', 'w') as f:
        f.write('\n'.join(y_test))
    with open(main_dir + main_folder_path+'test/masks.txt', 'w') as f:
        f.write('\n'.join(mask_test))
    # val data
    with open(main_dir + main_folder_path + 'val/input.txt', 'w') as f:
        f.write('\n'.join(x_val))
    with open(main_dir + main_folder_path+'val/output.txt', 'w') as f:
        f.write('\n'.join(y_val))
    with open(main_dir + main_folder_path+'val/masks.txt', 'w') as f:
        f.write('\n'.join(mask_val))


def main():
    print('mode: ',_argparse().mode)
    print('name: ',_argparse().config_name)
    mode = _argparse().mode
    data_file_name = _argparse().config_name
    fol_pic_name = _argparse().fol_img
    mask_path = _argparse().mask
    if(mode == 'per'):
        perspective_ds(fol_pic_name,mask_path,data_file_name)
    elif(mode == 'cube'):
        cube_ds(fol_pic_name,mask_path,data_file_name)
        


if __name__ == '__main__':
    main()