import cv2
import time
import os
import HandTrackingModule as htm

#webcam setting
cap = cv2.VideoCapture(2)
hcam, wcam = 480, 640
cap.set(3, wcam)
cap.set(4, hcam)
#fps parameters
curTime =  0
prevTime = 0

#finger Images
folderPath = "Finger Images"
myList = os.listdir(folderPath)
myList.sort()
imgList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    imgList.append(image)

tipIds = [4, 8, 12, 16, 20]

detector = htm.handDetector(detectionConf = 0.7)

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:
        fingers = []
        #special case for thumb
        #Right hand or inverted left hand
        if lmList[ tipIds[4] ][1] < lmList[ tipIds[0] ][1]:
            if lmList[ tipIds[0] ][1] > lmList[ tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        # Right hand or inverted left hand
        else:
            if lmList[ tipIds[0] ][1] < lmList[ tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        #cases for remaining 4 fingers
        for id in range(1, 5):
            if lmList[ tipIds[id] ][2] < lmList[ tipIds[id] - 2 ][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        h, w, c = imgList[totalFingers].shape
        img[0:h, 0:w] = imgList[totalFingers]
        cv2.rectangle(img, (0, 300), (100, 450), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{str(int(totalFingers))}', (0, 425), cv2.FONT_HERSHEY_PLAIN, 10,(0, 0, 255), 5)

    #frame per second
    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow("WebCam", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()