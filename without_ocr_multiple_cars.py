import cv2
import time
import datetime
import math
from PIL import Image
import numpy as np
import time
import os
from mobilenet_classes import *

'''
python without_ocr_mutiple_cars.py --image images/car.jpg --yolo yolo-coco

Uses Mobilenet model to crop if multiple cars are there in picture
Input-Recieves an cv2 img frame
Output-Calls license_detection by sending it cropped image of vehicle from the image

'''

img = cv2.imread(input('Enter relative Image path : ')) 

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, cv2.THRESH_BINARY, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (5, 5), 0)
    ret3, th3 = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(img):
    filtered = cv2.adaptiveThreshold(img.astype(
        np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def license(img):
    print('licsense called')
    net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3-tiny.cfg", "yolo-coco/yolov3-tiny_last.weights")
    
    image = np.array(img)

    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    print(" {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                            0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(image, (x, y), (x + w, y + h),(0,0,255), 2)
            cropped = Image.fromarray(image[y:y+h, x:x+w])
            cropped.save('output/images/predictions_mobilenet.jpg')
            text = "{}: {:.4f}".format(
                'Number Plate', confidences[i]) 
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,255), 2)

    image = cv2.resize(image, (1280, 720))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    image = cv2.imread('output/images/predictions_mobilenet.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    gray = remove_noise_and_smooth(gray)
    filename = "output/images/tesearct.jpg"
    cv2.imwrite(filename, gray)
    # text = pytesseract.image_to_string(Image.open(filename))
    # print(text)
    cv2.imshow("cropped", image)
    cv2.imshow("cropped_preprocessed", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vehicle(img):
    tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'pbpb.pbtxt')
    a = datetime.datetime.now()
    rows, cols, channels = img.shape

    tensorflowNet.setInput(cv2.dnn.blobFromImage(
        img, size=(300, 300), swapRB=True, crop=False))
    networkOutput = tensorflowNet.forward()
    count = 0
    
    for detection in networkOutput[0, 0]:
        score = float(detection[2])
        if score > 0.4:
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


vehicle(img)