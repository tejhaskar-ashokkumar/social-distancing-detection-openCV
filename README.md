# social-distancing-detection-openCV
Run main.py
Before running main, clone the YOLO pre-trained model from https://github.com/pjreddie/darknet

The system takes in the video as input. Initially, the people in the video are detected using a pre-trained YOLOv3 model. 
Once the people in the video are detected, the Euclidean distance between each nearest people are calculated, 
and if the distance is lesser than two meter, then it is considered safe, else it is considered unsafe.
