import shutil
from glob import glob
import argparse


def Move_Rename():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='C:/Users/Student/Downloads/train_imgs_labels/images/', help='Source_path')
    parser.add_argument('-c', type=int, default=1, help='Classification')
    parser.add_argument('-p', type=int, default=0.8, help='Proportion(0.8 = 8:2)')
    parser.add_argument('-d1', type=str, default='C:/git_workspace/train_valid_test/train/images/', help='New destination1')
    parser.add_argument('-d2', type=str, default='C:/git_workspace/train_valid_test/valid/images/', help='New destination2')
    opt = parser.parse_args()
    
    img_path = glob(opt.s + '*')
    img_num = len(img_path)
    img_proportion = int(img_num * 0.8)
    
    for i in range(img_num):
        img_name = img_path[i].split('images\\')[1]
        new_img_name = str(opt.c) + '_' + img_name
        if i <= img_proportion:
            shutil.copy(str(opt.s) + img_name, str(opt.d1) + new_img_name)
        else:
            shutil.copy(str(opt.s) + img_name, str(opt.d2) + new_img_name)
    
        
    #shutil.move(source_path_file, destination_path) 剪下再貼上
    #shutil.copy(source_path_file, destination_path) 複製貼上
    #shutil.copytree(source_path, destination_path) 複製目錄, destiantion_path不可存在
if __name__ == '__main__':
    Move_Rename()