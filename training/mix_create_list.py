from tqdm import tqdm
import argparse
import os


def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str, default='', help="percent to mix combine|least ")
    parser.add_argument("--want_config", "-want_config",nargs='+', default=[], help="folder name to store config dataset")
    parser.add_argument("--new_conf_name", "-new_conf_name",type=str, default='', help="folder that store dataset")
    arg = parser.parse_args()
    return arg

def create_fol(path):
    if not os.path.exists(path): 
        os.makedirs(path+'/test')
        os.makedirs(path+'/train')
        os.makedirs(path+'/val')
    else:
        print('already create '+path)

def main():
    main_dir = ""
    list_config = _argparse().want_config
    new_config_name = _argparse().new_conf_name
    create_fol(main_dir+'car_ds/train_test_config/'+new_config_name)

    type_file = ['train','test','val']
    type_label = ['input','output','masks']
    for file in type_file:
        for label in type_label:
            new_path = []
            all_path = []
            all_num_path = []
            for i in range(len(list_config)):
                each_path = []
                each_num_path = 0
                with open(main_dir+'car_ds/train_test_config/'+list_config[i]+'/'+file+'/'+label+'.txt') as f:
                    for count, line in enumerate(tqdm(f)):
                        each_path.append(line.strip())
                        each_num_path = each_num_path + 1
                all_path.append(each_path)
                all_num_path.append(each_num_path)

            minpos = all_num_path.index(min(all_num_path))
            # print(all_num_path)
            # print(minpos)
            # print(list_config[minpos])
            

            if(_argparse().mode == "least"):
                for j in range(len(all_path)):
                    # print(len(all_path[j][:all_num_path[minpos]]))
                    new_path = new_path + all_path[j][:all_num_path[minpos]]
            elif(_argparse().mode == "combine"):
                for j in range(len(all_path)):
                    new_path = new_path + all_path[j][:]


            main_folder_path = 'car_ds/train_test_config/'+new_config_name+'/'
            with open(main_dir + main_folder_path +file+'/'+label+'.txt', 'w') as f:
                f.write('\n'.join(new_path))

            print('-----------'+file+'/'+label+'.txt'+'---------------')
            print(len(new_path))  
            # print(new_path[0])
            # print(new_path[len(new_path)-1])
            print('----------------------------------------------------')


if __name__ == '__main__':
    main()