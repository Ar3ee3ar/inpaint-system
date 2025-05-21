
from PIL import Image

def get_file_name(path_list,mode):
    name_list = []
    for i in range(len(path_list)):
        # print(path_list[i])
        # input img process
        if(mode == "folder"):
            split_name_input = path_list[i].split('\\') # split path name
            name_list.append(split_name_input[1])
        elif(mode == "pic"):
            split_name_input = path_list[i].split('/')
            name_list.append(split_name_input[-1])
    return name_list