import os


def create_folders(path='/Users/leo/git_workspace/'):
    
    if not os.path.exists(path + 'train_valid_test'):
        os.chdir(path)
        os.mkdir(path + 'train_valid_test')
        os.mkdir(path + 'train_valid_test/train')
        os.mkdir(path + 'train_valid_test/train/images') 
        os.mkdir(path + 'train_valid_test/train/labels')
        
        os.mkdir(path + 'train_valid_test/valid')
        os.mkdir(path + 'train_valid_test/valid/images') 
        os.mkdir(path + 'train_valid_test/valid/labels')
        
        os.mkdir(path + 'train_valid_test/test')
        os.mkdir(path + 'train_valid_test/test/images') 
        os.mkdir(path + 'train_valid_test/test/labels')


if __name__ == '__main__':
    create_folders()
