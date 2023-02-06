import cv2
from glob import glob
import os
import numpy as np
from typing import Union, Tuple
import math

#1 圖片讀取 (多資料夾底下全部目錄, 圖檔)
def image_path_AllFolder(file_path):
    path_arr = []
    dir_name = []
    for dirpath, dirname, filenames in os.walk(file_path):
        if dirname:
            dir_name = dirname
        for filename in filenames:
            path_arr.append(os.path.join(dirpath, filename))
    return path_arr, dir_name
    # return dirname (目錄)
            

#2 圖片位置 (單資料夾底下全部圖檔)
def image_path(file_path):
    return glob(file_path + '/*')


#3 圖片讀取/高斯模糊/雙邊濾波/翻轉/存檔
def image_read(file_path, color=0, gaussian=0, bilateral=0, flip=0, flip_direction=-1, show=0, save=0, save_path='C:\git_workspace\img_processing_path\images', save_name='flip-img'):
    count = 0
    for i in file_path:
        count += 1
        img_type = i.split('.')[1]
        img = cv2.imread(i, color)
        if gaussian:
            img = cv2.GaussianBlur(img, (7, 7), 0)
        if bilateral:
            img = cv2.bilateralFilter(img, 7, 31, 31)
        if flip:
            img = cv2.flip(img, flip_direction) 
            # flip 1 水平翻轉, 0 垂直翻轉, -1 水平+垂直翻轉
        if save:
            os.chdir(save_path)
            cv2.imwrite(save_name + f'-{count:03d}.' + img_type, img)
        if show:
            cv2.imshow('image', img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""
  
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


if __name__ == '__main__':
    # color, gaussian, bilateral, flip, flip_direction, save, save_path, save_name
    # filp_direction 1 水平翻轉, 0 垂直翻轉, -1 水平+垂直翻轉
    
    #1 test
    #print(image_path_AllFolder(file := input('Input path:')))

    #3 test
    #image_read(image_path(file := input('input image path: ')), color=1, bilateral=1, flip=1, flip_direction=1, save=1)

    #_norm test
    img = cv2.imread('/Users/leo/git_workspace/train_valid_test/train/images/0_LINE_ALBUM_221229_0.jpg')
    with open('/Users/leo/git_workspace/train_valid_test/train/labels/0_LINE_ALBUM_221229_0.txt', 'r') as label1:
        l1 = label1.readline().split(' ')
    # with open('/Users/leo/Downloads/00108.txt', 'r') as label2:
    #     l2 = label2.readline().split(' ')
    
    start1 = _normalized_to_pixel_coordinates(float(l1[1]), float(l1[2]), 480, 640)
    end1 = _normalized_to_pixel_coordinates(float(l1[1])+float(l1[3]), float(l1[2])+float(l1[4]), 480, 640)
    # start2 = _normalized_to_pixel_coordinates(float(l2[1]), float(l2[2]), 640, 480)
    # end2 = _normalized_to_pixel_coordinates(float(l2[1])+float(l2[3]), float(l2[2])+float(l2[4]), 640, 480)
    
    cv2.rectangle(img, start1, end1, (255, 0, 0), 2)
    # cv2.rectangle(img, start2, end2, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()