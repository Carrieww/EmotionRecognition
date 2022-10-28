import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
# from keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import  load_model
# import matplotlib.pyplot as plt
import numpy as np

# load model
model = load_model("best_model.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture("Videos/raw/sample3.mov")

mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1,color=(0,128,0))

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    results = faceMesh.process(gray_img)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(test_img,faceLms,
            landmark_drawing_spec=drawSpec,connection_drawing_spec=drawSpec)
            mpDraw.draw_landmarks(
                    image=test_img,  
                    landmark_list=faceLms,
                    connections=mpFaceMesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawSpec,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255,255,255), thickness=5)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows