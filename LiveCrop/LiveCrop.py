import cv2
import numpy as np
import time
import os

flag = 0
save = 0
ix, iy, fx, fy = 0, 0, 10, 5
path = './saveimg/'
frameno = 0
if not os.path.exists('./saveimg'):
    os.mkdir('./saveimg/')

def note_coor(event, x, y, flags, param):
    global ix, iy, fx, fy, flag, save
    if event == cv2.EVENT_LBUTTONDOWN:
        ix , iy = x, y
    if event == cv2.EVENT_LBUTTONUP:
        fx, fy = x, y
        flag = 1

def empty(x):
    pass

cv2.namedWindow("WebCam")
cv2.resizeWindow("WebCam", 1024, 1024)
cv2.createTrackbar("Brightness", "WebCam", 116, 255, empty)
cv2.createTrackbar("Contrast", "WebCam", 28, 255, empty)
cv2.createTrackbar("Saturation", "WebCam", 98, 255, empty)
cap = cv2.VideoCapture(0)
cv2.namedWindow("WebCam")
while True:
    ret, frame = cap.read()
    img = frame
    if flag:
        flag = 0
        crop = img[min(iy,fy): max(fy,iy) , min(ix,fx): max(fx,ix)]
        fname = path + str(int(time.time()) % 1000) + '.png'
        cv2.imwrite(fname, crop)
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 0, 255), 1)

    Bright = cv2.getTrackbarPos("Brightness", "WebCam")
    Cont = cv2.getTrackbarPos("Contrast", "WebCam")
    Sat = cv2.getTrackbarPos("Saturation", "WebCam")

    cap.set(10, Bright)
    cap.set(11, Cont)
    cap.set(12, Sat)

    frameno += 1
    cv2.setMouseCallback("WebCam", note_coor)
    cv2.imshow("WebCam", img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()