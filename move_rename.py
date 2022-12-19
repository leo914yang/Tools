import shutil
from glob import glob

def Move_Rename(path='C:/git_workspace/train_valid_test'):
    file_path = glob(path + '/*')
    for h, i in enumerate(file_path):
        new_path = i.split('\\')[0] + f'/img-{h:04d}.' + i.split('.')[1]
        shutil.move(i, new_path)
        #print(new_path)

if __name__ == '__main__':
    Move_Rename(path := input('Input: '))