import torch
from dataset import CliffDataset
from torch.utils.data import DataLoader
from model import Yoloreo
from predictor import YoloreoPredictor
from utils import save_image_using_label,image_to_label_path

BASE_PATH = "/share/projects/cicero/checkpoints_baki/"

model_config = {
    "arch": "yolov8.yaml",
    "checkpoint":BASE_PATH+"weights_0/best.pt"
}

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## model def
model = Yoloreo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])
model.nc = 1
model.names = {0:'erosion'}

for k, v in model.named_parameters():v.requires_grad = False

model.to(device)


detector = YoloreoPredictor(cfg="default.yaml",csv_path="pred_split.csv",model=model)
detector.predict(save_res=False,create_shape_file=True)

"""
import pandas as pd
ddf = pd.read_csv("csv/image_valid_split2.csv")

for i in range(425,len(ddf)):
    path = ddf.iloc[i]["patch1"]
    save_image_using_label(path,image_to_label_path(path),f"test{i}.png")

"""
