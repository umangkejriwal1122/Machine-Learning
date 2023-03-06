from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)
detector = FaceDetector()
i = 0
while True:
    success, img = cap.read()
    myimg, bboxs = detector.findFaces(img)
    if bboxs:
        #i = i+1
        # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        #bbox = bboxs[0]["bbox"]
        #single_image = img[bbox[1]:bbox[1]-5 + bbox[3]+5, bbox[0]-5:bbox[0]+5 + bbox[2]+5]
        #cv2.imwrite("img"+str(i)+".jpg",single_image)
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        cv2.imshow("Image", single_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
