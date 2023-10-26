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

#model = YOLO('yolov8m.pt')
#ret =model(image)
#ret[0].save_txt("res.txt",True)
#print(20*"*")


class MyDetectionModel(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=False):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""

        self.first_forward = True
        super().__init__(cfg=cfg,ch=ch,nc=None,verbose=verbose)

        self.backbone = self.model[:10]
        self.head = self.model[10:]



    """
    method used to build the stride
    explanation: the super class call one forward with a lambda x to calculate the strides of the network
    file: ultralytics/nn/task.py
    line: 246
    """
    def _build_stride(self, x, profile=False, visualize=False):
        self.first_forward = False
        x = super()._predict_once(x, profile, visualize)
        return x

    def _predict_once(self, x, profile=False, visualize=False):

        ## refer to _build_stride comment
        if self.first_forward:
            return self._build_stride(x,profile,visualize)

        y, dt = [], []  # outputs


        features = self._forward_backbone(x)
        feature_1 = features[0:1]
        feature_2 = features[1:2]

        #feature_2 = self._forward_backbone(x[0])
        print("feature 1 ",feature_1.shape)
        print("feature 2 ",feature_2.shape)
        cross_attention_f1 = self._cross_attention(feature_1,feature_2)
        cross_attention_f2 =  self._cross_attention(feature_2,feature_1)


        """
        for m in self.backbone:
            print(m.f)
            if m.f != -1:  # if not from previous layer
                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        """
        """
        for m in self.head:
            if m.f != -1:  # if not from previous layer
                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        """
        return x


    def _forward_backbone(self,x):
        earlier_layer_output = []
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = earlier_layer_output[m.f] if isinstance(m.f, int) else [x if j == -1 else earlier_layer_output[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            earlier_layer_output.append(x if m.i in self.save else None)  # save output
        return x


    """
    For each element in feature_1, we check how it interacts with each element in feature_2.
    This interaction is determined by the attention scores,
    The attention scores quantify the relationship or compatibility between each element in feature_1 and each element in feature_2
    """
    def _cross_attention(self,feature_1,feature_2):

        scores = torch.matmul(feature_1, feature_2.transpose(-2, -1))  # Dot product
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attended_feature_2 = torch.matmul(attention_weights, feature_2)

        return attended_feature_2




image2 = cv2.imread("image.jpeg")
image2 = cv2.resize(image2, (640, 640))
image2 = image2.astype(np.uint8)
image2 = ToTensor()(image2)
image2 = image2.unsqueeze(0)
the_image = torch.cat([image,image2],dim=0)


theModel = MyDetectionModel(cfg="yolo8m.yaml")
theModel.load(torch.load('yolov8m.pt'))
ret = theModel(the_image)

#theModel.eval()
#print(image.shape)

#predictor = DetectionPredictor()
#x = predictor(source=[image,image], model=theModel)
#x[0].save_txt("res2.txt",True)
