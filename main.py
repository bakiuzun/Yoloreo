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

width, height, channels = 640, 640, 3

# Create a random array
image = cv2.imread("image.jpeg")
image = cv2.resize(image, (640, 640))
image = image.astype(np.uint8)
image = ToTensor()(image)
image = image.unsqueeze(0)


model = YOLO('yolov8m.pt')
ret =model(image)
ret[0].save_txt("res.txt",True)
print(20*"*")


class MyDetectionModel(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=False):  # model, input channels, number of classes
           """Initialize the YOLOv8 detection model with the given config and parameters."""
           super().__init__(cfg=cfg,ch=ch,nc=None,verbose=verbose)

    def _predict_once(self, x, profile=False, visualize=False):

        y, dt = [], []  # outputs

        backbone = self.model[:10]
        head = self.model[10:]

        for m in backbone:
            if m.f != -1:  # if not from previous layer
                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        for m in head:
            if m.f != -1:  # if not from previous layer
                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x




theModel = MyDetectionModel(cfg="yolo8m.yaml")
theModel.load(torch.load('yolov8m.pt'))
theModel.eval()
#ret = theModel(image)

predictor = DetectionPredictor()
x = predictor(source=image, model=theModel)
x[0].save_txt("res2.txt",True)
