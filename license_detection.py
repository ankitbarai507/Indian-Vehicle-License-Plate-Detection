import cv2
import numpy as np
import time
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

'''
Input-Cv2 image of license plate
Output-text within the image of license plate 
Dependency- Pytesseract OCR
'''
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
    text = pytesseract.image_to_string(Image.open(filename))
    print(text)
    cv2.imshow("cropped", image)
    cv2.imshow("cropped_preprocessed", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
