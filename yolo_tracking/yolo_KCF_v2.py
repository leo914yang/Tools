import cv2
import torch
import numpy as np
import math
from deepface import DeepFace
from glob import glob
import shutil
import os

def _iou(x1, y1, w1, h1, x2, y2, w2, h2):
    w = abs(max(x1, x2) - min(x1+w1, x2+w1))
    h = abs(max(y1, y2) - min(y1+h2, y2+h2))
    area = w * h
    area_x1 = w1 * h1
    area_x2 = w2 * h2
    iou = area / (area_x1 + area_x2 - area)
    return iou


def arr_sort(a):
    new_arr = np.sort(a)
    order = []
    for i in new_arr:
        ind = np.where(a == i)[0][0]
        order.append(ind)
    return order


def _coordinates(x, y):
    return math.floor(x), math.floor(y)


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
    
    # tracker coordinate for last loop; x1, y1, x2, y2
    tracker_coordinate = []

    # tracker init: x1, y1, x2, y2
    init_coordinate = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        #imgh = image.shape[0]
        #imgw = image.shape[1]

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

        # More than one high iou means our main character was blocked by someone else
        iou_of_yolo_and_tracker = []
  
        for index, i in enumerate(order):
            x1, y1 = _coordinates(xmin[i], ymin[i])
            x2, y2 = _coordinates(xmax[i], ymax[i])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(index), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 1, cv2.LINE_AA)
            init_coordinate.append([x1, y1, x2, y2])  
            if tracker_coordinate:
                iou_score = _iou(
                    int(tracker_coordinate[0]), int(tracker_coordinate[1]),
                    int(tracker_coordinate[2]), int(tracker_coordinate[3]), 
                    int(xmin[i]), int(ymin[i]), int(xmax[i])-int(xmin[i]), 
                    int(ymax[i])-int(ymin[i]))
                
                if iou_score > 0.7:
                    print('iou: ', iou_score)
                    iou_of_yolo_and_tracker.append(i)
                    main = [x1, y1, x2, y2]
            
        if tracking:
            success, point = tracker.update(image)
            if success:
                p1 = _coordinates(point[0], point[1])
                p2 = _coordinates(point[0] + point[2], point[1] + point[3])

                # save tracker coordinate: x1, y1, x2, y2
                tracker_coordinate = [point[0], point[1], point[0] + point[2], point[1] + point[3]]

                cv2.rectangle(image, p1, p2, (255, 0, 255), 2)

                # save 1 picture for our main character in images/
                _image = image[point[1]:point[1]+point[3], point[0]:point[0]+point[2]]
                pic_list = glob(path + '*')
                if len(pic_list) < 1:     
                    cv2.imwrite(path + str(len(pic_list)) + '.jpg', _image)  

                if len(pic_list) == 1 and not iou_of_yolo_and_tracker:
                    img2 = cv2.imread(pic_list[0], 1)
                    score = DeepFace.verify(image, img2, model_name='Facenet', distance_metric='cosine', model=deep_model, enforce_detection=False, detector_backend='ssd')
                    print('Main charactor move or out of edge:\n', score)
                    if not score['verified']:
                        tracker.init(image, main[0])

        # main character coordinate init; x1, y1, x2, y2
        main = []
        cv2.imshow('tracking', image)
        if cv2.waitKey(1) & 0xFF == 27:
            shutil.rmtree(path)
            os.mkdir(path)
            break

        if cv2.waitKey(1) & 0xFF == 48 and init_coordinate:
            tracker.init(image, init_coordinate[0])
            tracking = True

    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    #for i in webcam_input():
    #print(i)

    webcam_input()