""""
under development
Uzun Baki
"""
from model import *
from torch.utils.data import DataLoader
from dataset import CliffDataset
from trainer import MyDetectionTrainer


model_config = {
    "arch": "myyolov8m.yaml",
    "checkpoint":"imported/yolov8m.pt"
}

## model def
model = MyYolo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])
model.nc = 1
model.names = {0:"erosion"}


trainer = MyDetectionTrainer(cfg="cfg.yaml",model=model)
trainer.train()
