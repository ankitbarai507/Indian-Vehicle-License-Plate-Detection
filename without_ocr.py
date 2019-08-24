import argparse
import numpy as np
import argparse
import time
import cv2
import os
from PIL import Image
'''
USAGE
python without_ocr.py --image images/car.jpg
'''
ap = argparse.ArgumentParser()
# construct the argument parse and parse the arguments
    
ap.add_argument("-i", "--image",default='images/car.jpg', #required=True to make it mandatory
            help="path to input image")
ap.add_argument("-y", "--yolo", default='yolo-coco',
            help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
            help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
            help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

def yolo_license(args):
   
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                            dtype="uint8")
    weightsPath = os.path.sep.join([args["yolo"], "yolov3-tiny_last.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3-tiny.cfg"])
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    image = cv2.imread(args["image"])
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


    for output in layerOutputs:
  
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > args["confidence"]:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cropped = Image.fromarray(image[y:y+h, x:x+w])
            cropped.save('output/'+str(args['image']).split('.')[0]+'_predictions.jpg')
            text = "{}: {:.4f}".format(
                'Number Plate', confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
    image = cv2.resize(image, (1280, 720))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    image = cv2.imread('output/'+str(args['image']).split('.')[0]+'_predictions.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    gray = remove_noise_and_smooth(gray)
    filename = 'output/'+str(args['image']).split('.')[0]+"_preprocessed.jpg"
    cv2.imwrite(filename, gray)
    # text = pytesseract.image_to_string(Image.open(filename))
    # print(text)
    cv2.imshow("cropped", image)
    cv2.imshow("cropped_preprocessed", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



yolo_license(args)