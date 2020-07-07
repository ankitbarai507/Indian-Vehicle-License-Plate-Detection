import cv2
import time
import datetime
import math
from PIL import Image
import numpy as np
from mobilenet_vehicle_detection import *
import sys


'''
python mutiple_cars.py images/car.jpg
'''

img_path=sys.argv[1]
img = cv2.imread(img_path) 
vehicle(img)


