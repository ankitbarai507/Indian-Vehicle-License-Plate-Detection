import cv2
import time
import datetime
import math
from PIL import Image
import numpy as np
from mobilenet_vehicle_detection import *

'''
python mutiple_cars.py 
images\car.png
'''
img = cv2.imread(input('Enter relative Image path : ')) 
vehicle(img)


