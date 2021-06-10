import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
prevTime = 0

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence = 0.8)

while True :
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    imgHeight, imgWidth, _ = img.shape

    if results.detections :
        for id, detection in enumerate(results.detections) :
            # Bounding box from class(normalized values)
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * imgWidth), int(bboxC.ymin * imgHeight), \
                   int(bboxC.width *imgWidth), int(bboxC.height * imgHeight)

            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

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