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

class HypParameters:
    def __init__(self, hyp_dict):
        self.__dict__.update(hyp_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = {
    "arch": "deneme.yaml",
    "checkpoint":"yolov8m.pt"
}
train_config = {
    "batch_size":10,
    "epoch":5,
    "shuffle":True,
    "mode":"train",
}

hyper_parameter = {
    "lr0": 0.00001,  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
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
validation_config = {
    "batch_size":30,
    "shuffle":True,
    "mode":"validation",
}


hyp = HypParameters(hyper_parameter)

## model def
model = MyYolo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])
model.args = hyp


model.to(device)
## dataset def
train_dataset = CliffDataset(mode=train_config["mode"])
train_loader =  DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=train_config["shuffle"])

#validation_dataset = CliffDataset(mode=validation_config["mode"])
#validation_loader =  DataLoader(train_dataset, batch_size=validation_config["batch_size"], shuffle=validation_config["shuffle"])


## loss
criterion_head1 = v8DetectionLoss(model)
criterion_head2 = v8DetectionLoss(model)

parameters = model.parameters()

optimizer = torch.optim.Adam(
    parameters,
    lr=hyper_parameter['lr0'],
    weight_decay=hyper_parameter['weight_decay'],
)

pbar = tqdm(train_loader)


scaler = GradScaler()


for epoch in range(train_config['epoch']):
    model.train()
    total_loss = 0
    for idx, sample in enumerate(pbar):
        with autocast():
            features = model(sample["img"].to(device))
            patch_1_annotation,patch_2_annotation = train_dataset.retrieve_annotation(sample,device)

            loss, loss_items  = criterion_head1(features["x_1"],patch_1_annotation)
            print("LOSS = ",loss)


        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()


"""
epoch = 100
for epoch in range(epoch):
    # Model Training
    #model.train()

    pbar = tqdm(train_loader)

    for idx, sample in enumerate(pbar):
        print(sample.shape)

"""
