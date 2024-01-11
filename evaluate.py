"""
author Uzun Baki
-- u.officialdeveloper@gmail.com
"""

import torch
from validator import YoloreoValidator
from dataset import CliffDataset
from torch.utils.data import DataLoader
from model import Yoloreo

BASE_PATH = "/share/projects/cicero/checkpoints_baki/"

model_config = {
    "arch": "yolov8.yaml",
    "checkpoint":BASE_PATH+"weights_1/best.pt"
}

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## model def
model = Yoloreo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])
model.nc = 1
model.names = {0:'erosion'}
# freeze
for k, v in model.named_parameters():v.requires_grad = False

model.to(device)

validation_dataset = CliffDataset(path="csv/image_valid_split.csv");

validation_loader =  DataLoader(validation_dataset ,batch_size=32,shuffle=False)
validator = YoloreoValidator(dataloader=validation_loader,dataset=validation_dataset)

validator.evaluate(model,device,conf=0.25)
