import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

hCam = 480
wCam = 640
frameR = 100  #Frame reduction
smoothening = 3
prevlocX, prevlocY = 0, 0
curlocX, curlocY = 0, 0
prevTime = 0
wScreen, hScreen = autopy.screen.size()

cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands = 1)

while True :
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #Find Hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw = False)

    #Finger Moving Area
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 255), 2)

    #Get the tip of the index and middle finger
    if len(lmList) :
        _, x1, y1 = lmList[8]
        _, x2, y2 = lmList[12]

        #Check which fingers are up
        fingers = detector.fingersUp()
        fingersCount = fingers.count(1)
        if fingers[1] == 1 and fingers[2] == 0 and fingersCount == 1:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScreen))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))

            curlocX = prevlocX + (x3 - prevlocX) / smoothening
            curlocY = prevlocY + (y3 - prevlocY) / smoothening

            #Move mouse to (x3, y3)
            autopy.mouse.move(curlocX, curlocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            prevlocX = curlocX
            prevlocY = curlocY


        if fingers[1] == 1 and fingers[2] == 1 and fingersCount == 2 :
            length, img, lineInfo = detector.findDistance(8, 12, img)
            length = int(length)
            if length < 55 :
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                #Left Click
                autopy.mouse.click()


    curTime = time.time()
    fps = 1 / (curTime - prevTime)  # Frame per sec
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()