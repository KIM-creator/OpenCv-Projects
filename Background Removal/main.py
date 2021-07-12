import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
winWidth = 480
winHeight = 640
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", winHeight, winWidth)

imgIdx = 0
listBg = os.listdir("Images")
imgList = []
for ImgPath in listBg:
    imgBg = cv2.imread(f'Images/{ImgPath}')
    imgBg = cv2.resize(imgBg, (winHeight, winWidth))
    imgList.append(imgBg)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgOut = segmentor.removeBG(img, imgList[imgIdx], threshold = 0.7)

    # imgStacked = cvzone.stackImages([img, imgOut], 2, 1)

    # _, imgStacked = fpsReader.update(imgStacked)
    cv2.imshow("Image", imgOut)
    key = cv2.waitKey(1)
    if key == ord('a'):
        imgIdx = (imgIdx - 1) % len(imgList)
    elif key == ord('d'):
        imgIdx = (imgIdx + 1) % len(imgList)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()