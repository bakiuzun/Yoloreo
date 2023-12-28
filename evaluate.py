import torch
from validator import MyDetectionValidator
from dataset import CliffDataset
from torch.utils.data import DataLoader
from model import MyYolo

BASE_PATH = "/share/projects/cicero/checkpoints_baki/"
BASE_PATH = ""
model_config = {
    "arch": "yolov8.yaml",
    "checkpoint":BASE_PATH+"weights_0/best.pt"
}

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## model def
model = MyYolo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])
model.nc = 1
model.names = {0:'erosion'}


for k, v in model.named_parameters():v.requires_grad = False


model.to(device)


validation_dataset = CliffDataset(mode="valid")

validation_loader =  DataLoader(validation_dataset ,batch_size=32,shuffle=False)
validator = MyDetectionValidator(dataloader=validation_loader,dataset=validation_dataset)

validator.evaluate(model,device)
