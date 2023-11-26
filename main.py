""""
under development
Uzun Baki
"""
from model import *
from torch.utils.data import DataLoader
from dataset import CliffDataset
from trainer import MyDetectionTrainer


model_config = {
    "arch": "yolov8.yaml",
    "checkpoint":"imported/yolov8l.pt"
}

## model def
model = MyYolo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])
model.nc = 1
model.names = {0:"erosion"}

trainer = MyDetectionTrainer(cfg="cfg.yaml",model=model)
trainer.train()


"""
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from ultralytics.models.yolo.detect import DetectionPredictor

predictor = DetectionPredictor()


image = np.array(cv2.imread("image.jpg", cv2.IMREAD_UNCHANGED))
image = cv2.resize(image, (640, 640))
image = image.astype(np.uint8)
image = ToTensor()(image)
image = image.unsqueeze(0)


x = predictor(source=image ,model=model)
x[0].save_txt("res_brom",True)
"""
