""""
under development
Uzun Baki
"""
from model import *
from torch.utils.data import DataLoader
from dataset import CliffDataset
from trainer import MyTrainer


model_config = {
    "arch": "myyolov8m.yaml",
    "checkpoint":"imported/yolov8m.pt"
}

## model def
model = MyYolo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])
model.nc = 1
model.names = ["erosion"]


train_dataset = CliffDataset(mode="train")
trainer = MyTrainer(cfg="cfg.yaml",model=model,dataset=train_dataset)
trainer.train()
