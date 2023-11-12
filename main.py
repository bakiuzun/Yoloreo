""""
under development
Uzun Baki
"""
from model import *
from torch.utils.data import DataLoader
from dataset import CliffDataset
from tqdm import tqdm
from ultralytics.utils.loss import v8DetectionLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from trainer import MyTrainer
class HypParameters:
    def __init__(self, hyp_dict):
        self.__dict__.update(hyp_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = {
    "arch": "myyolov8m.yaml",
    "checkpoint":"yolov8m.pt"
}


hyper_parameter = {
    "lr0": 0.1,  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": 0.01,  # (float) final learning rate (lr0 * lrf)
    "momentum": 0.937,  # (float) SGD momentum/Adam beta1
    "weight_decay": 0.0005,  # (float) optimizer weight decay 5e-4
    "warmup_epochs": 3.0,  # (float) warmup epochs (fractions ok)
    "warmup_momentum": 0.8,  # (float) warmup initial momentum
    "warmup_bias_lr": 0.1,  # (float) warmup initial bias lr
    "box": 7.5,  # (float) box loss gain
    "cls": 0.5 , # (float) cls loss gain (scale with pixels)
    "dfl": 1.5  # (float) dfl loss gain
}


hyp = HypParameters(hyper_parameter)

## model def
model = MyYolo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])
model.args = hyp
model.to(device)


criterion = v8DetectionLoss(model)
## dataset def
train_dataset = CliffDataset(mode="train")
train_loader =  DataLoader(train_dataset, batch_size=5, shuffle=True)
trainer = MyTrainer(cfg="cfg.yaml",model=model,dataset=train_dataset)
trainer.train()
"""
parameters = model.parameters()

optimizer = torch.optim.Adam(
    parameters,
    lr=hyper_parameter['lr0'],
    weight_decay=hyper_parameter['weight_decay'],
)

pbar = tqdm(train_loader)
scaler = GradScaler()
"""
"""
pbar = tqdm(train_loader)



for idx, sample in enumerate(pbar):

    features = model(sample["img"].to(device))
    patch_1_annotation,patch_2_annotation = train_dataset.retrieve_annotation(sample,device)

    loss, loss_items  = criterion(features["x_1"],patch_1_annotation)
    print("LOSS = ",loss)
    break


"""
