from utils import  save_image_using_label,pred_one_image
from dataset import MAX_MIN
from model import MyYolo

image_path = "/share/projects/cicero/objdet/dataset/CICERO_stereo/images/1_Varengeville_sur_Mer/202107221109551_202107221110373/patches_img1/tiles_202107221109551_18560_05440.png"
label_path = "/share/projects/cicero/objdet/dataset/CICERO_stereo/train_label/1_Varengeville_sur_Mer/202107221109551_202107221110373/patches_cm_indiv_stereo/patches_cm1_txt/tiles_202107221109551_18560_05440.txt"

#save_image_using_label(image_path=image_path,label_path=label_path,save_path=None)




image_path = "/share/projects/cicero/objdet/dataset/CICERO_stereo/images/1_Varengeville_sur_Mer/202107221109551_202107221110373/patches_img1/tiles_202107221109551_33280_02880.png"




## model def
model = MyYolo(cfg="myyolov8m.yaml")
model.load_pretrained_weights("/share/projects/cicero/checkpoints_baki/cross_lr_0.0005_epoch_80.pt")
model.nc = 1
model.names = ["erosion"]
#pred_one_image(model,image_path,MAX_MIN,mode="train",output_file=None)

"""

import pandas as pd
from utils import image_to_label_path


df = pd.read_csv("csv/image_train_split.csv")
for i in range(15):
    image_path = df.iloc[i]["patch1"]
    print(image_path)
    pred_one_image(model,image_path,MAX_MIN,mode="validation",output_file=None)

    label_path = image_to_label_path(image_path,True)

    #save_image_using_label(image_path=image_path,label_path=label_path,save_path=f"test_{i}.png")



"""
