import cv2
import torch
import numpy as np
import math
from deepface import DeepFace
from glob import glob
import shutil
import os


# 計算兩個框框的iou
def _iou(x1, y1, w1, h1, x2, y2, w2, h2):
    w = abs(max(x1, x2) - min(x1+w1, x2+w2))
    h = abs(max(y1, y2) - min(y1+h1, y2+h2))
    area = w * h
    area_x1 = w1 * h1
    area_x2 = w2 * h2
    iou = area / (area_x1 + area_x2 - area)
    return iou


# array排序並回傳順序
def arr_sort(a):
    new_arr = np.sort(a)
    order = []
    for i in new_arr:
        ind = np.where(a == i)[0][0]
        order.append(ind)
    return order


# 座標取整數
def _coordinates(x, y):
    return math.floor(x), math.floor(y)


def _find_new_box(image, img2, iou_of_yolo_and_tracker, deep_model):
    max = 0
    list = {}
    for i in iou_of_yolo_and_tracker:
        # image = image[
        #     iou_of_yolo_and_tracker[i][0]:iou_of_yolo_and_tracker[i][2],
        #     iou_of_yolo_and_tracker[i][1]:iou_of_yolo_and_tracker[i][3]
        #     ]
        score = DeepFace.verify(
            image, img2, model_name='Facenet', 
            distance_metric='cosine', model=deep_model, 
            enforce_detection=False, detector_backend='ssd'
            )
        if not max or max < score['threshold']:
            list = {}
            list[score['threshold']] = iou_of_yolo_and_tracker[i]
            print(list)
            max = score['threshold']
    if max:
        return list[max]
    

def webcam_input():
    cap = cv2.VideoCapture(0)
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/Users/leo/Downloads/yolo_v5/yolov5/runs/train/exp3/weights/best.pt', force_reload=True)
    
    # kalman filter tracker
    tracker = cv2.TrackerCSRT_create()
    tracking = False

    # Deepface pre-build model
    deep_model = DeepFace.build_model('VGG-Face')

    # Path to save and read verify image
    path = '/Users/leo/git_workspace/Tools/images/'

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        #imgh = image.shape[0]
        #imgw = image.shape[1]

        main = []
        yolo = model(image)
        df = yolo.pandas().xyxy[0]
        mask = df['name'].eq('person')
        mask = df['confidence'] > 0.6
        #print(df[mask])
        xmin = df[mask]['xmin'].values
        ymin = df[mask]['ymin'].values
        xmax = df[mask]['xmax'].values
        ymax = df[mask]['ymax'].values
            
        order = arr_sort(xmin)
        #main = [int(x1), int(y1), int(abs(x1-x2)), int(abs(y1-y2))]
        
        # More than one high iou means our main character was blocked by someone else
        # {iou:[x1, y1, x2, y2]}
        iou_of_yolo_and_tracker = {}

        for index, i in enumerate(order):
            x1, y1 = _coordinates(xmin[i], ymin[i])
            x2, y2 = _coordinates(xmax[i], ymax[i])

            #if not tracking:   
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(index), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 1, cv2.LINE_AA)
            main.append([x1, y1, abs(x1 - x2), abs(y1 - y2)])
            
            if tracking:
                success, point = tracker.update(image)
                if success:
                    p1 = _coordinates(point[0], point[1])
                    p2 = _coordinates(point[0] + point[2], point[1] + point[3])

                    iou_score = _iou(p1[0], p1[1], point[2], point[3], int(xmin[i]), int(ymin[i]), int(xmax[i])-int(xmin[i]), int(ymax[i])-int(ymin[i]))
                    if iou_score > 0.6 and len(iou_of_yolo_and_tracker) < 2:
                        iou_of_yolo_and_tracker[iou_score] = [x1, y1, x2, y2]
                    # iou_of_yolo_and_tracker has max of 2 element
                    elif iou_score > 0.6 and len(iou_of_yolo_and_tracker) >= 2:
                        for i in iou_of_yolo_and_tracker:
                            if i < iou_score:
                                iou_of_yolo_and_tracker.pop(i)
                                iou_of_yolo_and_tracker[iou_score] = [x1, y1, x2, y2]
                                break
                    #center = _coordinates(point[0] + point[2]/2, point[1] + point[3]/2)
                    #if xmin[0] < p1[0] and ymin[0] < p1[1] and xmax[0] > p2[0] and ymax[i] > p2[1]:
                    cv2.rectangle(image, p1, p2, (255, 0, 255), 2)
                    

                    # save 1 picture for our main character
                    _image = image[point[1]:point[1]+point[3], point[0]:point[0]+point[2]]
                    pic_list = glob(path + '*')
                    
                    if len(pic_list) < 1:     
                        cv2.imwrite(path + str(len(pic_list)) + '.jpg', _image)
                    
                    if len(pic_list) == 1 and len(iou_of_yolo_and_tracker) > 1:
# here
                        img2 = cv2.imread(pic_list[0], 1)
                        new_coordinate = _find_new_box(image, img2, iou_of_yolo_and_tracker, deep_model)
                        if new_coordinate is not None:
                            print('Main character was blocked:\n', new_coordinate[0], new_coordinate[1], new_coordinate[2], new_coordinate[3])
                            tracker.init(image, [new_coordinate[0], new_coordinate[1], new_coordinate[2], new_coordinate[3]])

                    # elif len(pic_list) == 1 and iou_of_yolo_and_tracker and iou_of_yolo_and_tracker[0] < 0.3:
                    #     score = DeepFace.verify(image, img2, model_name='Facenet', distance_metric='cosine', model=deep_model, enforce_detection=False, detector_backend='ssd')
                    #     print('Main charactor move or out of edge:\n', score)
                    #     if not score['verified']:
                    #         tracker.init(image, [point[0], point[1], point[2], point[3]])

        cv2.imshow('tracking', image)
        if cv2.waitKey(1) & 0xFF == 27:
            shutil.rmtree(path)
            os.mkdir(path)
            break

        if cv2.waitKey(1) & 0xFF == 48 and main:
            tracker.init(image, main[0])
            tracking = True

    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    #for i in webcam_input():
    #print(i)

    webcam_input()