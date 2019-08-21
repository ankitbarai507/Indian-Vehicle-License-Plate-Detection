import cv2
import time
import datetime
import math
from PIL import Image
import numpy as np
import argparse
from yolo_license_detection import *
'''
USAGE
python mainfile.py --image images/car.jpg
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

yolo_license(args)

