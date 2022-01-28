import cv2
import numpy as np

#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#Load saved training data

name = {0 : "Trump",1 : "Bob"}

img = cv2.imread("test/biden.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_haar_cascade=cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')#Load haar classifier
faces_detected=face_haar_cascade.detectMultiScale(gray,1.3,5)#detectMultiScale returns rectangles
print(faces_detected)
for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        confidence = int(100-confidence)
        print("confidence:",confidence)
        print("label:",label)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=3)
        predicted_name=name[label]
        if confidence > 45:
           cv2.putText(img,predicted_name,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),4)

cv2.imshow("Detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()