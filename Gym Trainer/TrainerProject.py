import cv2
import numpy as np
import mediapipe as mp
import time
import PoseModule as pm

cap = cv2.VideoCapture("videos/biceps.mp4")
detector = pm.poseDetector()
count = 0
dir = "up"
curTime = 0
prevTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1080, 800))

    img = detector.findPose(img, False)
    lmlist = detector.findPosition(img, False)

    if len(lmlist) != 0 :
        #left arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (45, 150), (100, 0))

        #count the number of curls
        if per == 100 and dir == "up" :
                dir = "down"
        if per == 0 and dir == "down" :
                count += 1
                dir = "up"

        if per > 95 :
            barColor = (0, 255, 0)
        elif per >= 50 and per < 95 :
            barColor = (0, 175, 255)
        else :
            barColor = (0, 0, 255)

        cv2.rectangle(img, (0, 550), (250, 800), (0, 250, 0), cv2.FILLED)
        cv2.putText(img, str(count), (50, 750), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 15)
        cv2.putText(img, f'{int(per)}%', (850, 250), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        bar = int(np.interp(per, (0, 100), (790, 300)))
        cv2.rectangle(img, (950, 300), (1050, 790), barColor, 2)
        cv2.rectangle(img, (950, bar), (1050, 790), barColor, cv2.FILLED)


    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()