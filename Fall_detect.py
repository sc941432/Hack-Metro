from ultralytics import YOLO
import cv2
import os


os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO("fall_det_1.pt")

video_path = r"gaurav_fall.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True,conf=0.5)

        annotated_frame = results[0].plot()
        frame1=cv2.resize(annotated_frame,(600,800))

        cv2.imshow("YOLOv8 Tracking", frame1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:

        break

cap.release()
cv2.destroyAllWindows()
