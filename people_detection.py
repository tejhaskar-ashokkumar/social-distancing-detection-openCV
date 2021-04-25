# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:34:34 2020

@author: Ashok
"""

import cv2
import numpy as np

def detect_people(frame, pt_model, layer_names, label_index = 0 ):
    (H, W) = frame.shape[:2]
    results = []
    
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    pt_model.setInput(blob)
    layerOutputs = pt_model.forward(layer_names)
    
    boxes = []
    centroids = []
    confidences = []
    
    for i in layerOutputs:
        for j in i:
            scores = j[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if classID == label_index and confidence > 0.3:
                box = j[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                
    #applying non-maxima suppression to eliminate weak, overlapping bounding boxes
    nm_suppress = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
    
    if len(nm_suppress) > 0:
        for k in nm_suppress.flatten():
            (x, y) = (boxes[k][0], boxes[k][1])
            (w, h) = (boxes[k][2], boxes[k][3])
            
            r = (confidences[k], (x, y, x + w, y + h), centroids[k])
            results.append(r)
    
    return results
    
    
                
    
    
    