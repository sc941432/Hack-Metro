import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import math
import winsound
import pygame
import time
import numpy as np

model=YOLO('yolov8s.pt')
prev_sound_time = time.time()
new_sound_time = 0
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

pygame.init()
crash_sound = pygame.mixer.Sound("beep.mp3")
crash_sound1 = pygame.mixer.Sound("stay_away.mp3")

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
#                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids



tracker=Tracker()
count=0
cap=cv2.VideoCapture('trial.mp4')
down={}
up={}

counter_down=[]
counter_up=[]
all_rois = [None] * 100  # Initialize a list with 100 elements
temp_all_rois=[None] * 100
flag=int(0)
while True:
    new_sound_time = time.time()

    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
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
            #print(c)

       

    bbox_id=tracker.update(list)
    #print(bbox_id)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        red_line_x=550
        blue_line_x=175   
        offset = 7
        
        '''cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)'''

        ''' both lines combined condition . First condition is for red line'''
        ## condition for counting the cars which are entering from red line and exiting from blue line
        #if red_line_x < (cx + offset) and red_line_x > (cx - offset):
        
        if cx<=red_line_x and cx>blue_line_x:
            down[id]=cx
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

              # apply sound effect
            new_sound_time = time.time()
            if new_sound_time>=prev_sound_time+2:
                crash_sound1.play()
                prev_sound_time=new_sound_time
                for ids in range(100):
                    temp_all_rois[ids]=all_rois[ids]
                flag=1
                

            all_rois[id]=frame[y3:y4, x3:x4]
            
            for ids in range(100):
                if all_rois[ids] is not None:
                    cv2.imshow('ROI' + str(ids), all_rois[ids])

            if flag == 1:
                for ids in range(100):
                    if all_rois[ids] is not None:
                        if np.array_equal(all_rois[ids], temp_all_rois[ids]):
                            cv2.destroyWindow('ROI' + str(ids))
                            all_rois[ids] = None
                flag = 0


            cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

        
        '''if id in down:
           if blue_line_x < (cx + offset) and blue_line_x > (cx - offset):         
             cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
             cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
             cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
             #counter+=1
             counter_down.append(id)  # get a list of the cars and buses which are entering the line red and exiting the line blue'''
            

        # condition for cars entering from  blue line
        #if blue_line_x < (cx + offset) and blue_line_x > (cx - offset):
        if cx<=blue_line_x and cx>0:
            
            up[id]=cx
            
            if all_rois[id] is not None:
                cv2.destroyWindow('ROI' + str(id))
                all_rois[id]=None

            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
            cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) 

            crash_sound.play()
        '''if id in up:
           if red_line_x < (cx + offset) and red_line_x > (cx - offset):         
             cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
             cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
             #counter+=1
             counter_up.append(id)  # get a list of the cars which are entering the line 1 and exiting the line 2'''


        if cx>=red_line_x:

            
            if all_rois[id] is not None:
                cv2.destroyWindow('ROI' + str(id))
                all_rois[id]=None



    
    text_color = (255,255,255)  # white color for text
    red_color = (0, 0, 255)  # (B, G, R)   
    blue_color = (255, 0, 0)  # (B, G, R)
    green_color = (0, 255, 0)  # (B, G, R)  

    cv2.line(frame,(175,0),(175,600),blue_color,3)  # seconde line
    cv2.putText(frame,('blue line'),(175,268),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)  

    cv2.line(frame,(550,0),(550,600),red_color,3)  #  starting cordinates and end of line cordinates
    cv2.putText(frame,('red line'),(558,198),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
  

    '''downwards = (len(counter_down))
    cv2.putText(frame,('going down - ')+ str(downwards),(60,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1, cv2.LINE_AA)    

    
    upwards = (len(counter_up))
    cv2.putText(frame,('going up - ')+ str(upwards),(60,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)''' 

    cv2.imshow("frames", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
