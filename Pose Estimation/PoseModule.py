import cv2
import mediapipe as mp
import time

class poseDetector:

    def __init__(self, mode = False, complexity = True, smooth = True, detectionConf = 0.5, trackingConf = 0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.detectionConf, self.trackingConf)

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture(0)
    curTime = 0
    prevTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if lmList:
            cv2.circle(img, (lmList[26][1], lmList[26][2]), 20, (0, 255, 255), cv2.FILLED)

        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime

        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3)
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()