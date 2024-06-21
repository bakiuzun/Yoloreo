"""
author Uzun Baki
-- u.officialdeveloper@gmail.com
"""

import torch
from dataset import CliffDataset
from torch.utils.data import DataLoader
from model import Yoloreo
from predictor import YoloreoPredictor
from utils import save_image_using_label,image_to_label_path

BASE_PATH = "/share/projects/cicero/checkpoints_baki/"

model_config = {
    "arch": "yolov8.yaml",
    "checkpoint":BASE_PATH+"weights_17/best.pt"
}

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## model def
model = Yoloreo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])
model.nc = 1
model.names = {0:'erosion'}
for k, v in model.named_parameters():v.requires_grad = False
model.to(device)

# detector
detector = YoloreoPredictor(csv_path="pred_test_2_img_without_annot_0.csv",model=model,conf=0.50)
detector.predict(save_res=True,create_shape_file=True)
