import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
from PIL import Image
import math

def load_image(file_path):
    ## 4 band but were are taking only the rgb band for now 
    try:
        return np.array(Image.open(file_path))[:,:,:3]
    except:
        return np.full((640, 640,3), np.inf)


class CliffDataset(Dataset):
    def __init__(self,mode="train"):
        image_file = f"csv/image_{mode}_split.csv"
        self.dataframe = pd.read_csv(image_file)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,idx):
        
        # Access the row at the specified index in the DataFrame
        row = self.dataframe.iloc[idx]
        patch1 = row['patch1']
        patch2 = row['patch2']
        
        image_patch_1 = torch.tensor(load_image(patch1)).float().permute(2, 0, 1)
        image_patch_2 = torch.tensor(load_image(patch2)).float().permute(2, 0, 1)
        
        image_patch_1 = image_patch_1.unsqueeze(0)
        image_patch_2 = image_patch_2.unsqueeze(0)
        data = torch.cat([image_patch_1, image_patch_2], dim=0)
        return data



