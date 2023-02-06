import cv2
import time
import threading
import numpy as np
import math
import torch
from glob import glob
from deepface import DeepFace
import shutil
import os
import time

# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.Frame2 = []
        self.status = False
        self.isstop = False
        # kalman filter tracker
        self.tracker = cv2.TrackerCSRT_create() 
        self.tracking = False

	# 攝影機連接。
        self.capture = cv2.VideoCapture(URL)

    def set_Frame2(self, Frame2):
        self.Frame2 = Frame2
    
    def start(self):
	# 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()
    
    def stop(self):
        self.isstop = True
        print('ipcam stopped!')
    
    def verify_start(self):
        print('verify started!')
        threading.Thread(target=self.verify, daemon=True, args=()).start()

    def verify_stop(self):
        self.tracking = False
        print('verify stop!')

    def getframe(self):
	# 當有需要影像時，再回傳最新的影像。
        return self.Frame.copy()
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        
        self.capture.release()

    def verify(self):
        # verify fail counter
        counter = 0
        state = True
        while self.tracking:

            if not state:
                counter += 1
            else:
                counter = 0

            score = DeepFace.verify(
                self.Frame, self.Frame2, model_name='Facenet', 
                distance_metric='cosine', 
                enforce_detection=False, detector_backend='ssd'
                )
            print(score)
            state = score['verified']
            #if score['threshold'] < 0.4:
            
            if counter == 10:
                self.verify_stop()
            time.sleep(1)
        
    def track(self, coordinate):
        self.tracker.init(image, coordinate)
        self.tracking = True



def _coordinates(x, y):
    return math.floor(x), math.floor(y)

yolo_list = ['yolov5', 'yolov7']
model_list = [
    '/Users/leo/Downloads/yolo_v5/yolov5/runs/train/exp3/weights/best.pt',
    '/Users/leo/Downloads/yolov5l/weights/best.pt',
    '/Users/leo/git_workspace/yolov7/weights/best.pt',
    '/Users/leo/Downloads/yolov7-X-custom20/best.pt',
    '/Users/leo/Downloads/yolov5m/weights/best.pt',
    '/Users/leo/Downloads/yolov5s/weights/best.pt'
    ]

model = torch.hub.load('ultralytics/yolov5', 'custom',
    path=f'{model_list[1]}', force_reload=True)
    
# yolov7 load的格式不同
# model = torch.hub.load(
#     './yolov7', 'custom', source='local',
#     path_or_model=f'{model_list[2]}', force_reload=True
#     )

path = '/Users/leo/git_workspace/Tools/images/'
cap = ipcamCapture(0)

cap.start()
time.sleep(1)

start_time = time.time()
time_counter = 0
while True:
    image = cap.getframe()
    
    time_counter += 1
    fps = time.time() - start_time
    if fps != 0:
        cv2.putText(image, f'FPS:{time_counter/fps:.2f}', (100, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 1, cv2.LINE_AA)
    if cap.tracking:
        success, point = cap.tracker.update(image)
        if success:
            p1 = _coordinates(point[0], point[1])
            p2 = _coordinates(point[0] + point[2], point[1] + point[3])
                
            cv2.rectangle(image, p1, p2, (0, 0, 255), 3)
            cv2.putText(image, 'main', p1, cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 1, cv2.LINE_AA)
                
            # save 1 picture for our main character
            _image = image[point[1]:point[1]+point[3], point[0]:point[0]+point[2]]
            pic_list = glob(path + '*')
                    
            if len(pic_list) < 1:     
                cv2.imwrite(path + str(len(pic_list)) + '.jpg', _image)
            
            if len(pic_list) == 1 and cap.Frame2 == []:
                cap.set_Frame2(cv2.imread(pic_list[0], 1))
                cap.verify_start()
        
    else:
        yolo = model(image)
        df = yolo.pandas().xyxy[0]
        print(df)

        for index, i in enumerate(df.values):
            # 0: xmin, 1: ymin, 2: xmax, 3: ymax, 4: conf, 5: class, 6: name
            cv2.rectangle(image, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)
            cv2.putText(image, str(index), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, df['name'][index], (int(i[0])+100, int(i[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 1, cv2.LINE_AA)
            conf = df['confidence'][index]
            cv2.putText(image, f'{conf:.2f}', (int(i[0])+400, int(i[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 1, cv2.LINE_AA)
            if i[6] =='speaker':
                cap.track([int(i[0]), int(i[1]), int(abs(i[0]-i[2])), int(abs(i[1]-i[3]))])
            
    cv2.imshow('Image', image)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.verify_stop()
        cap.stop()
        shutil.rmtree(path)
        os.mkdir(path)
        break

   