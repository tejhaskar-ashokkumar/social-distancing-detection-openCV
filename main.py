# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:22:49 2020

@author: Ashok
"""

from people_detection import detect_people
from scipy.spatial import distance as d
import numpy as np
import cv2
import imutils
import os

USE_GPU = True
YOLO_PATH = r'C:\Users\Ashok\Desktop\CV_Practical\Social Distancing Detection System\yolo_coco'

labelsPath = os.path.sep.join([YOLO_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([YOLO_PATH, "yolov3.weights"])
configPath = os.path.sep.join([YOLO_PATH, "yolov3.cfg"])

print("Loading YOLO pre-trained model . . . ")
pt_model = cv2.dnn.readNet(weightsPath, configPath)

if USE_GPU:
    print("Activating CUDA . . . .")
    pt_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    pt_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
layer_names = pt_model.getLayerNames()
layer_names = [layer_names[n[0] - 1] for n in pt_model.getUnconnectedOutLayers()]
    
print("Loading Video Input . . . .")
videoPath = r'C:\Users\Ashok\Desktop\CV_Practical\Social Distancing Detection System\input\social_distancing.mp4'
video = cv2.VideoCapture(videoPath)
writer = None

while True:
    (cap, frame) = video.read()
    if not cap:
        break
    
    frame = imutils.resize(frame, width = 700)
    print('Calling people detection function . . . .')
    results = detect_people(frame, pt_model, layer_names, 
                            label_index = LABELS.index("person"))
    violate = set()
    
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        print('Calculating the distance between two persons')
        DIST = d.cdist(centroids, centroids, metric = "euclidean")
        
        for i in range(0, DIST.shape[0]):
            for j in range(i+1, DIST.shape[1]):
                if DIST[i, j] < 50:
                    violate.add(i)
                    violate.add(j)
                    
    for(i, (prob, bounding_box, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bounding_box
        (cX, cY) = centroid
        color = (0, 255, 0)
        if i in violate:
            color = (0, 0, 255)
        
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
        text = "People Violated: {}".format(len(violate))
        cv2.putText(frame, text,(10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
        
    
    
    
        
    
    
    
    
    
    

