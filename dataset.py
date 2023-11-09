import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
from utils import (load_image,image_to_label)


class CliffDataset(Dataset):
    def __init__(self,mode="train"):
        self.mode = mode
        sample_file = f"csv/image_{mode}_split.csv"
        target_file = f"csv/label_{mode}_split.csv"
        self.dataframe = pd.read_csv(sample_file)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,idx):
        
        # Access the row at the specified index in the DataFrame
        row = self.dataframe.iloc[idx]
        patch1 = row['patch1']
        patch2 = row['patch2']
        stereo = False if np.isnan(patch2) else True

        image_patch_1 = load_image(patch1)[:,:,:3] / .255
        image_patch_1 = torch.tensor(image_patch_1).float().permute(2, 0, 1)
        
        if stereo:
            image_patch_2 = load_image(patch2)[:,:,:3] / .255
            image_patch_2 = torch.tensor(image_patch_2).float().permute(2, 0, 1)
        else:
            image_patch_2 = image_patch_1.clone()
        
        image_patch_1 = image_patch_1.unsqueeze(0)
        image_patch_2 = image_patch_2.unsqueeze(0)

        data = torch.cat([image_patch_1, image_patch_2], dim=0)
        
        res = {"img":data,"stereo":stereo}
        
        return res
    def _get_label_file_info(self,row,stereo,index):
        """
        bounding box .....
        """
        patch1 = row['patch1']
        patch2 = row['patch2']
        label_1 = image_to_label(img_file=patch1,patch1=True)
        label_2 = label_1
        if stereo:
            label_2 = image_to_label(img_file=patch2,patch1=False)

        
        ##### 



