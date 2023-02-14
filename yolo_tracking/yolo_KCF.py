import cv2
import torch
import numpy as np
import math


def IOU(x1, y1, w1, h1, x2, y2, w2, h2):
    w = abs(max(x1, x2) - min(x1+w1, x2+w2))
    h = abs(max(y1, y2) - min(y1+h1, y2+h2))
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

# For webcam input:
def webcam_input():
    cap = cv2.VideoCapture(0)
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/Users/leo/Downloads/yolo_v5/yolov5/runs/train/exp3/weights/best.pt', force_reload=True)
    # kalman filter tracker
    tracker = cv2.legacy.TrackerCSRT_create()  # 創建追蹤器
    tracking = False
    # ASCII = [48, 49, 50, 51, 52]
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        imgh = image.shape[0]
        imgw = image.shape[1]

        if tracking:
            success, point = tracker.update(image)
            if success:
                p1 = _coordinates(point[0], point[1])
                p2 = _coordinates(point[0] + point[2], point[1] + point[3])
                
                cv2.rectangle(image, p1, p2, (0, 0, 255), 3)
                cv2.putText(image, 'main', p1, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 1, cv2.LINE_AA)
                
                yield [p1, p2]

           
        else:
            main = []
            yolo = model(image)
            df = yolo.pandas().xyxy[0]
            mask = df['name'].eq('person')
            mask = df['confidence'] > 0.6
            # print(df[mask])
            xmin = df[mask]['xmin'].values
            ymin = df[mask]['ymin'].values
            xmax = df[mask]['xmax'].values
            ymax = df[mask]['ymax'].values
            
            order = arr_sort(xmin)
            # main = [int(x1), int(y1), int(abs(x1-x2)), int(abs(y1-y2))]
            
            for index, i in enumerate(order):
                cv2.rectangle(image, (int(xmin[i]), int(ymin[i])), (int(xmax[i]), int(ymax[i]) ), (0, 255, 0), 2)
                cv2.putText(image, str(index), (int(xmin[i]), int(ymin[i])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 1, cv2.LINE_AA)
                main.append([int(xmin[i]), int(ymin[i]), int(abs(xmin[i]-xmax[i])), int(abs(ymin[i]-ymax[i]))])
            
            
        cv2.imshow('show image', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        if cv2.waitKey(1) & 0xFF == 48 and main:
            tracker.init(image, main[0])
            tracking = True

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    webcam_input()