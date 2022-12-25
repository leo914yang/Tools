import shutil
from glob import glob
import argparse


def Move_Rename():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='/Users/leo/Downloads/', help='Source_path')
    parser.add_argument('-c', type=int, default=0, help='Classification')
    parser.add_argument('-p', type=int, default=0.8, help='Proportion(0.8 = 8:2)')
    parser.add_argument('-d1', type=str, default='/Users/leo/git_workspace/train_valid_test/train/', help='New destination1')
    parser.add_argument('-d2', type=str, default='/Users/leo/git_workspace/train_valid_test/valid/', help='New destination2')
    opt = parser.parse_args()
    
    img_path = glob(opt.s + 'images/*')
    img_num = len(img_path)
    img_proportion = int(img_num * 0.8)

    lab_path = glob(opt.s + 'labels/*')
    print(img_num, len(lab_path))

    for i in range(img_num):
        img_name = img_path[i].split('images/')[1]
        new_img_name = str(opt.c) + '_' + img_name

        lab_name = ''.join(lab_path[i].split('labels/')[1].split('.png'))
        new_lab_name = str(opt.c) + '_' + lab_name

        if i <= img_proportion:
            shutil.copy(img_path[i], str(opt.d1) + 'images/' + new_img_name)
            shutil.copy(lab_path[i], str(opt.d1) + 'labels/' + new_lab_name)
        else:
            shutil.copy(img_path[i], str(opt.d2) + 'images/' + new_img_name)
            shutil.copy(lab_path[i], str(opt.d2) + 'labels/' + new_lab_name)
    
        
    #shutil.move(source_path_file, destination_path) 剪下再貼上
    #shutil.copy(source_path_file, destination_path) 複製貼上
    #shutil.copytree(source_path, destination_path) 複製目錄, destiantion_path不可存在
if __name__ == '__main__':
    Move_Rename()