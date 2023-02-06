import cv2
import Mediapipe_face_test as mt
def read_video_pbp(video_path, frame):
    cap = cv2.VideoCapture(video_path)
    # frame_rate代表每幾幀擷取一次
    frame_rate = 3
    count = 1

    while(True):
        ret, frame = cap.read()
        if ret:
            if count % frame_rate == 0:
                print("擷取影片第：" + str(count) + " 幀")
            # 將擷取圖片縮小，便於顯示
            resize_img = cv2.resize(frame, (540, 960), interpolation=cv2.INTER_AREA)
            cv2.imshow('frame', resize_img)
            cv2.waitKey(0)
            count += 1
        else:
            pass
    cap.release()
    cv2.destroyAllWindows()
    print('程式執行結束')
    

