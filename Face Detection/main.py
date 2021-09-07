import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(frame, detection)
            #print(id, detection)
            #print(detection.score)
            h, w, c = frame.shape
            #print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, bbox, (255,0,255), 2)
            cv2.putText(frame, f"{int(detection.score[0]* 100)}%", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()