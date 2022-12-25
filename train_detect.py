import os 
from glob import glob


def train():
    path='/Users/leo/git_workspace/yolov7/'
    batch = input('Input Batch: ')
    cfg = input('Input cfg(yolo): ') + '.yaml'
    epoch = input('Input Epochs: ')
    device = input('Input device: ')
    os.system(f"python {path + 'train.py'} --batch {batch} --cfg {path + 'cfg/training/' + cfg} --img 320 --epochs {epoch} --data {path + 'Teammate/data.yaml'} --weights '' --name yolov7_test --hyp {path + 'data/hyp.scratch.p5.yaml'} --device {device}")


def detect():
    path='/Users/leo/git_workspace/train_valid_test/test/images'
    file_path = glob(path + '/*')
    weight_path = input('Input weight_path: ')
    confidence = input('Input confidence: ')
    device = input('Input device: ')
    for i in file_path:
        os.chdir('/Users/leo/git_workspace/yolov7')
        os.system(f'python detect.py --weights runs/train/{weight_path}/weights/best.pt --conf {confidence} --device {device} --source {i}')


if __name__ == '__main__':
    list = ['train', 'detect']
    choose = int(input('Choose function: train(0) or detect(1)?'))
    if list[choose] == 'train':
        train()
    elif list[choose] == 'detect':
        detect()

