from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

emotion_classifier = load_model('/home/dani/PycharmProjects/Licenta/Emotion_train_model2.h5')
face_classifier = cv2.CascadeClassifier('/home/dani/Licenta/functions/haarcascade_frontalface_default.xml')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

video_input = cv2.VideoCapture(0)

while True:
    # Luam doar un singur frame din inputul video
    ret, frame = video_input.read()
    labels = []
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, 1.3, 6)

    for (x, y, w, h) in faces:
        #Incadram cu un dreptunghi de culoare verde fata detectata in imagine
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #Selectam regiunea dorita din imagine si o convertim in imagine gri de dimensiune 48x48
        roi_gray = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)


        #Daca se detecteaza vreo fata in inputul video, atunci o convertim in array pentru a putea fi clasificata
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        #Modelul face o predictie, iar in functie de predictia facuta se adauga in frame emotia recunoscuta

            preds = emotion_classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'Nu detectez nicio fata', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_input.release()
cv2.destroyAllWindows()