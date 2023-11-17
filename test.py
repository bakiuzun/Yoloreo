
import numpy as np
from torchvision.transforms import ToTensor
from ultralytics.nn.tasks import *
from ultralytics.models.yolo.detect import DetectionPredictor
import pandas as pd
from dataset import MAX_MIN
from utils import load_image
import os
import cv2
import torch
from model import MyYolo
import sys


def get_min_max_dataset(mode="train"):

    df = pd.read_csv(f"csv/image_{mode}_split.csv")
    le_max = -1
    le_min = sys.maxsize

    for i in range(len(df)):
        row = df.iloc[i]
        patch1 = row["patch1"]
        patch2 = row["patch2"]

        file = patch1
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = image[:,:,:3]
        max_1 = np.max(image)
        min_1 = np.min(image)
        if pd.isna(patch2):
            le_max = max(le_max,max_1)
            le_min = min(min_1,le_min)

        else:
            file2 = patch2
            image_2 = cv2.imread(file2, cv2.IMREAD_UNCHANGED)
            image_2 = image_2[:,:,:3]
            max_2 = np.max(image_2)
            min_2 = np.min(image_2)

            le_max = max(max_1,max_2,le_max)
            le_min = min(min_1,min_2,le_min)

    return le_max,le_min



def pred_one_image(image_file,ckpt,cfg="myyolov8m.yaml",mode="train"):

    model = MyYolo(cfg=cfg)
    model.load_pretrained_weights(ckpt)
    model.nc = 1
    model.names = ["erosion"]


    predictor = DetectionPredictor()

    image = load_image(image_file)
    image = image[:,:,:3]
    image = (image - MAX_MIN[f"{mode}_min"]) / (MAX_MIN[f"{mode}_max"] - MAX_MIN[f"{mode}_min"])

    image = torch.tensor(image).float().permute(2, 0, 1)
    image = image.unsqueeze(0)

    x = predictor(source=image ,model=model)
    for i in range(len(x)):
        x[i].save_txt(f"pred_res{i}.txt",True)
