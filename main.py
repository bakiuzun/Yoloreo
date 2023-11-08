""""
under development
Uzun Baki
"""
from model import *
from torch.utils.data import DataLoader
from dataset import CliffDataset
from tqdm import tqdm

## model def 
model = MyYolo(cfg="deneme.yaml")
model.load_pretrained_weights('yolov8m.pt')


batch_size = 32 
shuffle = True
train_dataset = CliffDataset(mode="train")
train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)


pbar = tqdm(train_loader)

for idx, sample in enumerate(pbar):
    model(sample)
    break


"""
epoch = 100
for epoch in range(epoch):
    # Model Training
    #model.train()

    pbar = tqdm(train_loader)

    for idx, sample in enumerate(pbar):
        print(sample.shape)

"""



#model = YOLO('yolov8m.pt')
#model.train(data="data.yaml")
#ret =model(image)
#ret[0].save_txt("res.txt",True)
#print(20*"*")




"""
theModel = MyDetectionModel(cfg="deneme.yaml")
theModel.load_pretrained_weights('yolov8m.pt')

#theModel.train()
predictor = DetectionPredictor()
x = predictor(source=the_image ,model=theModel)
x[0].save_txt("res1.txt",True)
x[1].save_txt("res1.txt",True)
"""