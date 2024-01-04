
import torch
from model import MyYolo
import torch
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from ultralytics.models.yolo.detect import DetectionPredictor
import pandas as pd
import torch
from model import MyYolo
from utils import (image_to_label_path,save_image_using_label,get_georeferenced_pos,min_max_norm)
import geopandas as gpd
from shapely.geometry import Polygon


model_config = {
    "arch": "myyolov8m.yaml",
    "checkpoint":"weights_2/best.pt" # change path
}

## load model
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyYolo(cfg=model_config["arch"])
model.load_pretrained_weights(model_config["checkpoint"])
model.nc = 1
model.names = {0:'erosion'}

for k, v in model.named_parameters():v.requires_grad = False
model.to(device)



# image dataset
df = pd.read_csv("image_valid_split.csv")

predictor = DetectionPredictor(cfg="default.yaml")

data = []

georef_poses = []

for i in range(len(df)):
    path = df.iloc[i]["patch1"]
    image = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED))[:,:,:3]
    image.astype("float")
    image = min_max_norm(image)
    image = ToTensor()(image)
    image = image.unsqueeze(0)

    #uncomment if you want to save the image with the true bounding box
    #label = image_to_label_path(path)
    #save_image_using_label(path,label,f"image{i}.png")

    x = predictor(source=image ,model=model)

    # for each predicted bounding box
    for pos in x[0].boxes.xyxy.cpu().numpy():
        #uncomment if you want to see the predicted bounding box on the image
        """
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = image[:,:,:3]
        image_max = np.max(image)
        image_min = np.min(image)
        image = ((image - image_min) / (image_max - image_min)) * 255
        print("Y 0 ",)

        cv2.rectangle(image, (int(y[0]), int(y[1])), (int(y[2]), int(y[3])), (0, 255, 0), 2)  # Draw rectangle
        cv2.imwrite(f"saha{i}.png", image)
        """
        # pos[0] = x1,  pos[1] = y1,  pos[2] = x2,  pos[3] = y2

        # get the georeferenced pos from two points and the image path
        xy = get_georeferenced_pos(path,int(pos[0]),int(pos[1]))
        xy2 = get_georeferenced_pos(path,int(pos[2]),int(pos[3]))

        georef_poses.append([xy[0],xy[1],xy2[0],xy2[1]])

    #uncomment if you want to save the coordinate in a txt file (x,y,center_x,center_y)
    #x[0].save_txt(f"{path[:path.rfind('.')]}.txt",True)

polygons = [Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]),(bbox[2], bbox[3]), (bbox[0], bbox[3])]) for bbox in georef_poses]

if len(polygons) > 0:
    print("LEN I ",len(polygons))
    print("LEN DF ",len(df))
    data = {'geometry': polygons}
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')  # Change EPSG code as needed

    # Save the GeoDataFrame to a shapefile
    output_shapefile = 'bounding_boxes.shp'
    gdf.to_file(output_shapefile)
