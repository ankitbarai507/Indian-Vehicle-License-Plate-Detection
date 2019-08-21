import cv2
import time
import datetime
import math
from PIL import Image
import numpy as np
from mobilenet_vehicle_detection import *

'''
python mutiple_cars.py --image images/car.jpg --yolo yolo-coco
'''
img = cv2.imread(input('Enter relative Image path : ')) 
vehicle(img)


