import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(2)                   #using external webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
curTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           #Hands class take only RGB image
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for HandLms in results.multi_hand_landmarks:
            for id, lm in enumerate(HandLms.landmark):
                h, w, c = img.shape                 #height, width, channel
                cx, cy = int(lm.x * w) , int(lm.y * h)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (250, 0, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, HandLms, mpHands.HAND_CONNECTIONS)


    curTime = time.time()
    fps = 1 / (curTime - prevTime)                          #Frame per sec
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()