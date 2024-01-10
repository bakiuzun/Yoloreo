"""
author Uzun Baki
-- u.officialdeveloper@gmail.com
"""

from model import *
from torch.utils.data import DataLoader
from dataset import CliffDataset
from trainer import YoloreoTrainer


config = {
    "arch": "yolov8.yaml",
    "checkpoint":"imported/yolov8m.pt",
    "train_path":"csv/image_train_split.csv",
    "valid_path":"csv/image_valid_split.csv",
}


## model def
model = Yoloreo(cfg=config["arch"])
model.load_pretrained_weights(config["checkpoint"])
model.nc = 1
model.names = {0:"erosion"}

trainer = YoloreoTrainer(cfg="cfg.yaml",train_path=config["train_path"],valid_path=config["valid_path"], model=model)
trainer.train()
