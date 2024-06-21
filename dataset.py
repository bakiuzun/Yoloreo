"""
author Uzun Baki
-- u.officialdeveloper@gmail.com
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
from utils import (load_image,image_to_label_path,get_label_info)
import copy
import random
import cv2
from utils import (load_image,image_to_label_path,get_label_info,make_augmentation,mixup_image)
import copy 
import albumentations as A

seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)

class CliffDataset(Dataset):
    """
    Cliff Dataset
    """
    def __init__(self,path,mode="train"):
        self.dataframe = pd.read_csv(path)
        self.mode = mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,idx):

        # Access the row at the specified index in the DataFrame
        row = self.dataframe.iloc[idx]
        patch1 = row['patch1']
        patch2 = row['patch2']
        # if patch 2 is  nan -> it's a mono image
        stereo = False if pd.isna(patch2) else True

        im_files_patch1 = patch1
        im_files_patch2 = patch1 if stereo == False else patch2

        image_patch_1 = load_image(patch1)[:,:,:3]
        #image_patch_1 = load_image(patch1)[:,:,:3]
        image_patch_1 = image_patch_1.astype("float")
        image_patch_1 = cv2.resize(image_patch_1, (640, 640))
        ## min max norm
        minn = np.min(image_patch_1)
        maxx = np.max(image_patch_1)

        image_patch_1 = ((image_patch_1 - minn) / (maxx - minn)) * 255
        image_patch_1 = ToTensor()(image_patch_1)

        if stereo:
            ## same process as the first image patch 1
            image_patch_2 = load_image(patch2)[:,:,:3]
            image_patch_2 = image_patch_2.astype("float")
            image_patch_2 = cv2.resize(image_patch_2, (640, 640))

            minn = np.min(image_patch_2)
            maxx = np.max(image_patch_2)
            image_patch_2 = ((image_patch_2 - minn) / (maxx - minn)) * 255
            image_patch_2 = ToTensor()(image_patch_2)

        else:
            # copy image 1 if it's not stereo
            image_patch_2 = copy.deepcopy(image_patch_1)

        image_patch_1 = image_patch_1.unsqueeze(0)
        image_patch_2 = image_patch_2.unsqueeze(0)
        data = torch.cat([image_patch_1, image_patch_2], dim=0)

        # data shape (batch,2,channels,width,height)
        # where 2 -> (image_patch_1, image_patch_2) where you can have stereo pair or image_patch_1 = image_patch_2 if stereo is False
        res = {"img":data,"stereo":stereo,"im_files_patch1":im_files_patch1,"im_files_patch2":im_files_patch2}

        return res

    def _get_label_file_info(self,patch_1_file,patch_2_file,stereo,index):

        """
        get the labels path from the images path and return the bounding box,cls,idx for each label
        if it's not stereo we copy the mono image
        """

        label_1_path = image_to_label_path(img_file=patch_1_file,patch1=True)
        label_1_info = get_label_info(label_1_path,index)

        if stereo == False:
            ## if not stereo copy the path of the mono image
            return label_1_info, copy.deepcopy(label_1_info)
        else:
            label_2_path = image_to_label_path(img_file=patch_2_file,patch1=False)
            label_2_info = get_label_info(label_2_path,index)
            return label_1_info,label_2_info


    def retrieve_annotation(self,batch,device):
        """
        method called during training to get the annotation (bounding box,cls,idx) from the image files
        """

        patch_1_annotation = {"bboxes":[],"cls":[],"batch_idx":[]}
        patch_2_annotation = {"bboxes":[],"cls":[],"batch_idx":[]}

        images = torch.empty(len(batch['img']),2, 3, 640, 640)

        for i in range(len(batch["img"])):
            label_1,label_2 = self._get_label_file_info(batch["im_files_patch1"][i],batch["im_files_patch2"][i],batch["stereo"][i],i)
            
            dif1 = 0
            dif2 = 0
            len_box_1 = len(label_1['bboxes'])
            len_box_2 = len(label_2['bboxes'])

            if self.mode == "train":
                
                img1 = batch["img"][i,0].cpu().numpy().transpose((1,2,0)) 
                img2 = batch["img"][i,1].cpu().numpy().transpose((1,2,0)) 
                
                #augs = random.choice(["MixUp",A.HorizontalFlip(p=1.),A.RandomRotate90(p=1.),A.Perspective(p=1.,fit_output=True)])
                augs = random.choice(["MixUp",A.HorizontalFlip(p=1.),A.RandomRotate90(p=1.),A.Perspective(p=1.,fit_output=True)])

                if augs  != "MixUp":
                    augmented_img1,augmented_img2,augmented_bbox1,augmented_bbox2 = make_augmentation(img1,img2,label_1["bboxes"],label_2["bboxes"],augs)

                    augmented_img1 = ToTensor()(augmented_img1)
                    augmented_img2 = ToTensor()(augmented_img2)

                    data = torch.cat([augmented_img1.unsqueeze(0), augmented_img2.unsqueeze(0)], dim=0)
                    images[i] = data

                    # Sometimes the perspective augmentation remove boxes
                    dif1 = len_box_1 - len(augmented_bbox1)    
                    dif2 = len_box_2 - len(augmented_bbox2)    
                    label_1["bboxes"] = augmented_bbox1
                    label_2["bboxes"] = augmented_bbox2

                if augs == "MixUp" and random.choice([1,2]) == 1:
                    next_img_index = (i + 1) % len(batch['img'])
                    next_label_1,next_label_2 = self._get_label_file_info(batch["im_files_patch1"][next_img_index],batch["im_files_patch2"][next_img_index],batch["stereo"][next_img_index],i)

                    next_img1 = batch["img"][next_img_index,0].cpu().numpy().transpose((1,2,0))
                    next_img2 = batch["img"][next_img_index,1].cpu().numpy().transpose((1,2,0))

                    mixed_img1 = mixup_image(img1,next_img1)
                    mixed_img2 = mixup_image(img2,next_img2)
                                        
                    augmented_img1 = ToTensor()(mixed_img1.astype(np.uint8))
                    augmented_img2 = ToTensor()(mixed_img2.astype(np.uint8))
                    data = torch.cat([augmented_img1.unsqueeze(0), augmented_img2.unsqueeze(0)], dim=0)
                    images[i] = data

                    label_1["bboxes"] = np.concatenate((label_1["bboxes"],next_label_1["bboxes"]),axis=0)
                    label_1["cls"] =  np.concatenate((label_1["cls"], next_label_1["cls"]), axis=0)
                    label_1["batch_idx"] = np.concatenate((label_1["batch_idx"],next_label_1["batch_idx"]),axis=0)   

                    label_2["bboxes"] = np.concatenate((label_2["bboxes"],next_label_2["bboxes"]),axis=0)
                    label_2["cls"] =  np.concatenate((label_2["cls"], next_label_2["cls"]), axis=0)
                    label_2["batch_idx"] = np.concatenate((label_2["batch_idx"],next_label_2["batch_idx"]),axis=0)   

                    len_box_1 = len(label_1["bboxes"])
                    len_box_2 = len(label_2["bboxes"])

                    
            patch_1_annotation["bboxes"].extend(label_1["bboxes"])
            patch_1_annotation["cls"].extend(label_1["cls"][:len_box_1-dif1])
            patch_1_annotation["batch_idx"].extend(label_1["batch_idx"][:len_box_1-dif1])

            patch_2_annotation["bboxes"].extend(label_2["bboxes"])
            patch_2_annotation["cls"].extend(label_2["cls"][:len_box_2-dif2])
            patch_2_annotation["batch_idx"].extend(label_2["batch_idx"][:len_box_2-dif2])


        patch_1_annotation['batch_idx'] = torch.tensor(patch_1_annotation['batch_idx']).to(device)
        patch_1_annotation['cls'] = torch.tensor(np.array(patch_1_annotation['cls'])).to(device)
        patch_1_annotation['bboxes'] = torch.tensor(np.array(patch_1_annotation['bboxes'])).to(device)

        patch_2_annotation['batch_idx'] = torch.tensor(patch_2_annotation['batch_idx']).to(device)
        patch_2_annotation['cls'] = torch.tensor(np.array(patch_2_annotation['cls'])).to(device)
        patch_2_annotation['bboxes'] = torch.tensor(np.array(patch_2_annotation['bboxes'])).to(device)

        if self.mode == "train":
            batch["img"] = images
            return patch_1_annotation,patch_2_annotation,batch

        return patch_1_annotation,patch_2_annotation
