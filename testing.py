from ultralytics import YOLO
import cv2
import time
import pandas as pd

# load yolov8 model
model = YOLO('yolov8s.pt')
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


# load video
#video_path = 'video2.webm'
cap = cv2.VideoCapture(0)
t1 = time.time()

while True:
    ret, frame = cap.read()
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    
    
    #frame = cv.resize(frame, None, fx=1, fy=1, interpolation=cv.INTER_AREA)

    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)

    a=results[0].boxes.data
    a = a.detach().cpu().numpy()  # added this line
    px=pd.DataFrame(a).astype("float")
    #print(px)

    list=[]
             
    for index,row in px.iterrows():
#        print(row) 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        #list.append([x1,y1,x2,y2])
        if 'person' in c:
            list.append([x1,y1,x2,y2])


    for (x, y, w, h) in list:
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        t2 = time.time()
        if (t2 - t1) > 1:
            print('I SEE YOU IN 1')
            t1 = time.time() # reset start time

    cv2.imshow('ROI1', frame)


    # Detect Motion
    # Modify frames to detect contours
    if 'person' in c:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around detected contours
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 1000:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), thickness=2)
            cv2.putText(frame1, 'Status: {}'.format('Movement'), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=3)
            
        # cv.drawContours(frame1, contours, -1, (0,255,0), thickness=2)

    # Show Motion Detected Feed
    cv2.imshow('Motion Feed', frame1)
  
    # Press ESC to break
    if cv2.waitKey(1)&0xFF==27:
        break
    '''c = cv2.waitKey(1)
    if c == 27:
        break'''

cap.release()
cv2.destroyAllWindows()

