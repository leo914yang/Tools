import os


def create_folders(path='C:/git_workspace/'):
    
    if not os.path.exists(path + 'train'):
        os.mkdir(path + 'train')
        os.mkdir(path + 'train/images') 
        os.mkdir(path + 'train/labels')
    if not os.path.exists(path + 'valid'):
        os.mkdir(path + 'valid')
        os.mkdir(path + 'valid/images') 
        os.mkdir(path + 'valid/labels')


if __name__ == '__main__':
    create_folders(path := input('Input: '))
