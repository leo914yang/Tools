import cv2
import mediapipe as mp
import os
import Image_processing as ip

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
def static_images():
  IMAGE_FILES = ['C:/git_workspace/yolov7/1001.png']
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  with mp_face_mesh.FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
      image = cv2.imread(file)
      imgwidth = image.shape[0]
      imgheight = image.shape[1]
      # Convert the BGR image to RGB before processing.
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print and draw face mesh landmarks on the image.
      if not results.multi_face_landmarks:
        continue
      annotated_image = image.copy()
      for face_landmarks in results.multi_face_landmarks:
        face_localtion(face_landmarks.landmark, imgwidth, imgheight)
        print('face_landmarks:', face_landmarks)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
      cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
def webcam_input():
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  cap = cv2.VideoCapture(0)
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      
      results = face_mesh.process(image)
      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()


def face_localtion(landmark, c, img_name, path):
  
  x = landmark.location_data.relative_bounding_box.xmin
  y = landmark.location_data.relative_bounding_box.ymin
  w = landmark.location_data.relative_bounding_box.width
  h = landmark.location_data.relative_bounding_box.height
  output = str(c) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
  with open(path + f'/{img_name}.txt', 'w') as w:
    w.write(output)


def rac_face(show_img=0):
  path = input('Img path: ')
  label_folder = input('Label folder path: ')
  classification = input('Classification: ')
  
  img = ip.image_path(path)
  mp_face_detection = mp.solutions.face_detection   # 建立偵測方法
  mp_drawing = mp.solutions.drawing_utils           # 建立繪圖方法
  for i in img:
    print(i)
    img_name = i.split('images\\')[1]
    img_file = cv2.imread(i)
    with mp_face_detection.FaceDetection(             # 開始偵測人臉
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        
        img2 = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)   # 將 BGR 顏色轉換成 RGB
            
        results = face_detection.process(img2)        # 偵測人臉
            
        if results.detections:
            for detection in results.detections:
              face_localtion(detection, classification, img_name, label_folder)
              mp_drawing.draw_detection(img_file, detection)  # 標記人臉
        if show_img:
          cv2.imshow('oxxostudio', img_file)
          cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=='__main__':
  # webcam_input()
  # static_images()
  # show_img 0 or 1
  # 先label再丟進train_test_valid
  rac_face(show_img=0)