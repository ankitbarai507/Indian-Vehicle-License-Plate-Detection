import cv2
import time
import datetime
import math
from PIL import Image
import numpy as np

import time
import os
from license_detection import *
from mobilenet_classes import *

'''
Uses Mobilenet model to crop if multiple cars are there in picture
Input-Recieves an cv2 img frame
Output-Calls license_detection by sending it cropped image of vehicle from the image

'''
def vehicle(img):
    tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'pbpb.pbtxt')
    a = datetime.datetime.now()
    rows, cols, channels = img.shape

    tensorflowNet.setInput(cv2.dnn.blobFromImage(
        img, size=(300, 300), swapRB=True, crop=False))
    networkOutput = tensorflowNet.forward()
    count = 0
    print('hi1')
    for detection in networkOutput[0, 0]:

        score = float(detection[2])
        if score > 0.4:
            print('hi')
            print(classes_90[int(detection[1])])
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.putText(img, classes_90[int(detection[1])], (int(
                left), int(top)-15), 0, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            if classes_90[int(detection[1])] in ["bus", "train", "truck","bicycle", "car", "motorcycle"]:
                count += 1
                cv2.rectangle(img, (int(left), int(top)), (int(right),
                                                        int(bottom)), (0, 0, 255), thickness=2)
                left = int(left)
                top = int(top)
                bottom = int(bottom)
                right = int(right)
                cropped = Image.fromarray(img[top:bottom, left:right])
                cropped.save('output/images/predictions1_cropped.jpg')
                license(cropped)
                
    cv2.imshow('Image_mobilenet_cropped', img)
    b = datetime.datetime.now()
    print('no of cars', count)
    cv2.waitKey(0)
    
    
