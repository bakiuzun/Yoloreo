import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
from utils import (load_image,image_to_label_path,get_label_info)
import copy


## MIN MAX CALCULATED WITHOUT THE 4th BAND
MAX_MIN = {
    "train_max":24377,
    "train_min":0
}



class CliffDataset(Dataset):
    def __init__(self,mode="train"):
        self.mode = mode
        sample_file = f"csv/image_{mode}_split.csv"
        self.dataframe = pd.read_csv(sample_file)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,idx):

        # Access the row at the specified index in the DataFrame
        row = self.dataframe.iloc[idx]
        patch1 = row['patch1']
        patch2 = row['patch2']
        stereo = False if pd.isna(patch2) else True

        im_files_patch1 = patch1
        im_files_patch2 = patch1 if stereo == False else patch2

        image_patch_1 = load_image(patch1)[:,:,:3]
        max_patch_1 = np.max(image_patch_1)
        min_patch_1 = np.min(image_patch_1)
        #image_patch_1 = (image_patch_1 - MAX_MIN[f"{self.mode}_min"]) / (MAX_MIN[f"{self.mode}_max"] - MAX_MIN[f"{self.mode}_min"])
        image_patch_1 = (image_patch_1 - min_patch_1) / (max_patch_1 - min_patch_1)
        image_patch_1 = torch.tensor(image_patch_1).float().permute(2, 0, 1)

        if stereo:
            image_patch_2 = load_image(patch2)[:,:,:3]
            max_patch_2 = np.max(image_patch_2)
            min_patch_2 = np.min(image_patch_2)
            #image_patch_2 = (image_patch_2 - MAX_MIN[f"{self.mode}_min"]) / (MAX_MIN[f"{self.mode}_max"] - MAX_MIN[f"{self.mode}_min"])
            image_patch_2 = (image_patch_2 - min_patch_2) / (max_patch_2 - min_patch_2)
            image_patch_2 = torch.tensor(image_patch_2).float().permute(2, 0, 1)
        else:
            image_patch_2 = copy.deepcopy(image_patch_1)

        image_patch_1 = image_patch_1.unsqueeze(0)
        image_patch_2 = image_patch_2.unsqueeze(0)

        data = torch.cat([image_patch_1, image_patch_2], dim=0)

        res = {"img":data,"stereo":stereo,"im_files_patch1":im_files_patch1,"im_files_patch2":im_files_patch2 }

        return res

    def _get_label_file_info(self,patch_1_file,patch_2_file,stereo,index):

        """
        bounding box .....
        """

        label_1_path = image_to_label_path(img_file=patch_1_file,patch1=True)
        label_1_info = get_label_info(label_1_path,index)

        if stereo == False:
            return label_1_info, copy.deepcopy(label_1_info)
        else:

            label_2_path = image_to_label_path(img_file=patch_2_file,patch1=False)
            label_2_info = get_label_info(label_2_path,index)
            return label_1_info,label_2_info


    def retrieve_annotation(self,batch,device):


        patch_1_annotation = {"bboxes":[],"cls":[],"batch_idx":[]}
        patch_2_annotation = {"bboxes":[],"cls":[],"batch_idx":[]}

        for i in range(len(batch["img"])):
            label_1,label_2 = self._get_label_file_info(batch["im_files_patch1"][i],batch["im_files_patch2"][i],batch["stereo"][i],i)

            patch_1_annotation["bboxes"].extend(label_1["bboxes"])
            patch_1_annotation["cls"].extend(label_1["cls"])
            patch_1_annotation["batch_idx"].extend(label_1["batch_idx"])

            patch_2_annotation["bboxes"].extend(label_2["bboxes"])
            patch_2_annotation["cls"].extend(label_2["cls"])
            patch_2_annotation["batch_idx"].extend(label_2["batch_idx"])


        patch_1_annotation['batch_idx'] = torch.tensor(patch_1_annotation['batch_idx']).to(device)
        patch_1_annotation['cls'] = torch.tensor(patch_1_annotation['cls']).to(device)
        patch_1_annotation['bboxes'] = torch.tensor(np.array(patch_1_annotation['bboxes'])).to(device)

        patch_2_annotation['batch_idx'] = torch.tensor(patch_2_annotation['batch_idx']).to(device)
        patch_2_annotation['cls'] = torch.tensor(patch_2_annotation['cls']).to(device)
        patch_2_annotation['bboxes'] = torch.tensor(np.array(patch_2_annotation['bboxes'])).to(device)

        return patch_1_annotation,patch_2_annotation
