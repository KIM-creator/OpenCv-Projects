import cv2
import mediapipe as mp
import time

class FaceDetection :
    def __init__(self, detectionConf = 0.8) :
        self.minDetectionConf = detectionConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConf)

    def findFaces(self, img, draw = True) :
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        imgHeight, imgWidth, _ = img.shape
        bboxs = []

        if self.results.detections :
            for id, detection in enumerate(self.results.detections) :
                # Bounding box from class(normalized values)
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * imgWidth), int(bboxC.ymin * imgHeight), \
                       int(bboxC.width *imgWidth), int(bboxC.height * imgHeight)
                bboxs.append([bbox, detection.score])
                if draw :
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return bboxs, img

    def fancyDraw(self, img, bbox, leng = 20, thick = 5):
        cv2.rectangle(img, bbox, (0, 255, 0), 1)
        x, y, h, w = bbox
        x1, y1 = x + w, y + h
        #top left corner
        cv2.line(img, (x, y), (x + leng, y), (0, 255, 0), thick)
        cv2.line(img, (x, y), (x, y + leng), (0, 255, 0), thick)

        #top right corner
        cv2.line(img, (x1 - leng, y), (x1, y), (0, 255, 0), thick)
        cv2.line(img, (x1, y), (x1, y + leng), (0, 255, 0), thick)

        #bottom left corner
        cv2.line(img, (x, y1), (x + leng, y1), (0, 255, 0), thick)
        cv2.line(img, (x, y1 - leng), (x, y1), (0, 255, 0), thick)

        #bottom right corner
        cv2.line(img, (x1 - leng, y1), (x1, y1), (0, 255, 0), thick)
        cv2.line(img, (x1, y1 - leng), (x1, y1), (0, 255, 0), thick)

        return img

def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    detector = FaceDetection()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        bboxs, img = detector.findFaces(img)

        #frame per second
        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime

        cv2.putText(img, f'{int(fps)}', (50, 80), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.imshow("Output", img)
        if cv2.waitKey(10) & 0xFF == ord('q') :
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__' :
    main()