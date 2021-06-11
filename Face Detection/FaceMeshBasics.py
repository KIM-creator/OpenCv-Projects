import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(2)
prevTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 5, min_detection_confidence = 0.75)
drawSpec = mpDraw.DrawingSpec(thickness = 2, circle_radius = 1)

while True :
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks :
        for faceLms in results.multi_face_landmarks :
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark) :
                h, w, _ = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                print(id, x, y)

    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    cv2.putText(img, f'{int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()