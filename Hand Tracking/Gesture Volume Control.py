import cv2
import time
import numpy as np
import math
import HandTrackingModule as htm
#pycaw module for volume control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


#parameters
camWidth, camHeight = 640, 480
curTime, prevTime = 0, 0
vol, volbar, volper = 0, 400, 0

#using external webcam
cap = cv2.VideoCapture(2)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = htm.handDetector(detectionConf = 0.7)

#volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw = False)

    if len(lmlist) != 0:
        #index finger and thumb
        idno, x1, y1 = lmlist[4]
        idno, x2, y2 = lmlist[8]
        cx , cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 15, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 5)
        #distance between these two points
        length = int(math.hypot((x2 - x1), (y2 - y1)))
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)

        vol = np.interp(length, [30, 250], [minVol, maxVol])
        volbar = np.interp(length, [30, 250],[400, 100])
        volper = np.interp(length, [30, 250], [0, 100])

    volume.SetMasterVolumeLevel(vol, None)
    cv2.rectangle(img, (20, 100), (50, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (20, int(volbar)), (50, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'% = {int(volper)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 0, 255), 2)

    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 0, 255),  2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()