""""
under development
Uzun Baki
"""
import torch
from ultralytics import YOLO
import torch.nn as nn
from ultralytics.nn.modules.head import Classify
import numpy as np
from torchvision.transforms import ToTensor
from ultralytics.nn.tasks import *
import cv2
from ultralytics.models.yolo.detect import DetectionPredictor
from utils import parse_my_detection_model


width, height, channels = 640, 640, 3

# Create a random array
image = cv2.imread("image.jpeg")
image = cv2.resize(image, (640, 640))
image = image.astype(np.uint8)
image = ToTensor()(image)
image = image.unsqueeze(0)

image2 = cv2.imread("image.jpeg")
image2 = cv2.resize(image2, (640, 640))
image2 = image2.astype(np.uint8)
image2 = ToTensor()(image2)
image2 = image2.unsqueeze(0)
the_image = torch.cat([image,image2],dim=0)

#model = YOLO('yolov8m.pt')
#ret =model(image)
#ret[0].save_txt("res.txt",True)
#print(20*"*")




from model import *

theModel = MyDetectionModel(cfg="deneme.yaml")
theModel.load_pretrained_weights('yolov8m.pt')

predictor = DetectionPredictor()
x = predictor(source=the_image ,model=theModel)
x[0].save_txt("res1.txt",True)
x[1].save_txt("res1.txt",True)
