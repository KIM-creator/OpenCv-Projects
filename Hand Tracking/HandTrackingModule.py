import cv2
import mediapipe as mp
import time

class handDetector():
    def __int__(self, mode = False, maxHands = 2,
                detectionConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        # self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConf, self.trackConf)
        # self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # Hands class take only RGB image

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                           self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handno = 0, idno = 0, draw = True):
        lmlist = []

        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]

            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape                     # height, width, channel
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw and id == idno:
                    cv2.circle(img, (cx, cy), 15, (250, 0, 0), cv2.FILLED)

        return lmlist


def main():
    cap = cv2.VideoCapture(2)                       # using external webcam
    prevTime = 0
    curTime = 0
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)

        curTime = time.time()
        fps = 1 / (curTime - prevTime)              # Frame per sec
        prevTime = curTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()