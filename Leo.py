import os 
from glob import glob

path = 'python detect.py --weights runs/train/yolov74/weights/best.pt --conf 0.1 --device cpu --source '
img_path = glob('Aquarium/test/images/*')

for i in img_path:
	os.system(path + i)

x = 123