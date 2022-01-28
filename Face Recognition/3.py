import cv2
import numpy as np


#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#Load saved training data

name = {0 : "Umang",1 : "Biden"}

cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    face_haar_cascade=cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')#Load haar classifier
    faces_detected=face_haar_cascade.detectMultiScale(gray_img,1.3,5)#detectMultiScale returns rectangles

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        confidence = int(100-confidence)
        print("confidence:",confidence)
        print("label:",label)
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)
        predicted_name=name[label]
        if confidence > 45:
           cv2.putText(test_img,predicted_name,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),4)

    cv2.imshow('face recognition tutorial ',test_img)
    
    if cv2.waitKey(1) == 27:    #escape key
        break

cap.release()
cv2.destroyAllWindows()

