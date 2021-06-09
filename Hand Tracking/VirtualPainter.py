import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

# Brush and eraser Parameters
brushColor = (255, 0, 255)
brushThickness = 15
eraserThickness = 100

cap = cv2.VideoCapture(0)
prevTime = 0
curTime = 0

#Header images
folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
imgCanvas = np.zeros((480, 640, 3), np.uint8)

for imgPath in myList :
    img = cv2.imread(f'{folderPath}/{imgPath}')
    img = img[90: 300, 0:]
    img = cv2.resize(img, (640, 100))
    overlayList.append(img)

header = overlayList[1]
xp, yp = 0, 0
detector = htm.handDetector(detectionConf = 0.85)

while True :
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:
        # tip of index and middle fingers
        _, x1, y1 = lmList[8]
        _, x2, y2 = lmList[12]

        fingers = detector.fingersUp()
        if fingers[1] and fingers[2] :
            #Selection Mode
            # xp, yp = x1, y1
            if y1 < 100 :
                if x1 >= 20 and x1 <= 140:
                    header = overlayList[0]
                    brushColor = (0, 0, 0)
                elif x1 >= 220 and x1 <= 300:
                    header = overlayList[2]
                    brushColor = (255, 0, 0)
                elif x1 >= 380 and x1 <= 460:
                    header = overlayList[3]
                    brushColor = (0, 0, 255)
                elif x1 >= 520 and x1 <= 610:
                    header = overlayList[1]
                    brushColor = (255, 0, 255)
        elif fingers[1] and fingers[2] == False:
            #Drawing Mode
            cv2.circle(img, (x1, y1), 10, brushColor, cv2.FILLED)

            if xp == 0 and yp == 0 :
                xp, yp = x1, y1

            if brushColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), brushColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), brushColor, eraserThickness)
            else :
                cv2.line(img, (xp, yp), (x1, y1), brushColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), brushColor, brushThickness)

        xp, yp = x1, y1

    #merging Canvas Image and original image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    #setting the header image
    img[0 : 100, 0 : 640] = header

    #frame per sec
    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    cv2.putText(img, f'{int(fps)}', (40, 400), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()