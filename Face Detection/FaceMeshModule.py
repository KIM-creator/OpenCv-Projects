import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, max_faces = 5, detectionConf = 0.75, thick = 2, radius = 1) :
        self.max_faces = max_faces
        self.detectionConf = detectionConf
        self.thick = thick
        self.radius = radius
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces = self.max_faces, min_detection_confidence = self.detectionConf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = self.thick, circle_radius = self.radius)


    def findFaceMesh(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []

        if self.results.multi_face_landmarks :
            for faceLms in self.results.multi_face_landmarks :
                if draw :
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark) :
                    h, w, _ = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    face.append([x, y])
                faces.append(face)

        return img, faces


def main() :
    cap = cv2.VideoCapture(2)
    prevTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img, faces = detector.findFaceMesh(img)

        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime

        cv2.putText(img, f'{int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Output", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__" :
    main()