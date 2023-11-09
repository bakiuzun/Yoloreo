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

model_config = {
    "arch": "deneme.yaml",
    "checkpoint":"yolov8m.pt"
}
train_config = {
    "batch_size":32,
    "epoch":20,
    "shuffle":True,
    "mode":"train",
    "lr0": 0.01,  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": 0.01,  # (float) final learning rate (lr0 * lrf)
    "momentum": 0.937,  # (float) SGD momentum/Adam beta1
    "weight_decay": 0.0005,  # (float) optimizer weight decay 5e-4
    "warmup_epochs": 3.0,  # (float) warmup epochs (fractions ok)
    "warmup_momentum": 0.8,  # (float) warmup initial momentum
    "warmup_bias_lr": 0.1  # (float) warmup initial bias lr
}

validation_config = {
    "batch_size":32,
    "shuffle":True,
    "mode":"validation",
}


## model def 
model = MyYolo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])

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
    lr=train_config['learning_rate'],
    betas=train_config['adam_betas'],
    weight_decay=train_config['weight_decay'],
)

pbar = tqdm(train_loader)

"""
for epoch in range(train_config['epoch']):
    model.train()
    for idx, sample in enumerate(pbar):
        features = model(sample["data"])
        loss = criterion_head1(features,)
        break

    optimizer.zero_grad()
    criterion_head1.backward()
    optimizer.step()
"""

"""
epoch = 100
for epoch in range(epoch):
    # Model Training
    #model.train()

    pbar = tqdm(train_loader)

    for idx, sample in enumerate(pbar):
        print(sample.shape)

"""