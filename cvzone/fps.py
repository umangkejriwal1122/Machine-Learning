import cvzone
import cv2

fpsReader = cvzone.FPS()
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    fps, img = fpsReader.update(img,pos=(50,80),color=(0,255,0),scale=5,thickness=5)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()