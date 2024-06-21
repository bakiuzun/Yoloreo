"""
author Uzun Baki
-- u.officialdeveloper@gmail.com
"""

import contextlib
import torch
import torch.nn as nn
import cv2
from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3,
                                    RTDETRDecoder, Segment)
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import (make_divisible)
import numpy as np
import pandas as pd
import sys
from ultralytics.models.yolo.detect import DetectionPredictor
import sys
from osgeo import gdal
import random
import albumentations as A


BASE_LABEL_FILE_PATH = "/share/projects/cicero/objdet/dataset/CICERO_stereo/train_label/1_Varengeville_sur_Mer/"
BASE_IMG_FILE_PATH = "/share/projects/cicero/objdet/dataset/CICERO_stereo/images/1_Varengeville_sur_Mer/"

BASE_IMG_FILE_PATH = '/share/projects/cicero/objdet/dataset/CICERO_stereo/images/3_Zakynthos/'
BASE_LABEL_FILE_PATH = "/share/projects/cicero/objdet/dataset/CICERO_stereo/train_label/3_Zakynthos/"

seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)


def image_to_label_path(img_file,patch1=True):
    """
    get the label file from the image path
    """
    img_file = img_file.split("/")
    patch_name = "patches_cm1_txt" if patch1 else "patches_cm2_txt"

    # tiles_201802171130571_13440_09920.PNG, -> tiles_201802171130571_13440_09920.txt
    img_file[-1] = img_file[-1].replace('.PNG', '.txt')

    label_path = BASE_LABEL_FILE_PATH + img_file[9] +  "/patches_cm_indiv_stereo/" + patch_name + "/" + img_file[-1]

    return label_path



def get_label_info(path,index):
    """
    from the label path, parameter:path
    read the content and store bounding box, classes and the batch index
    the batch index come from the paramter:index
    """
    bboxes = np.empty((0, 4))  # Initialize bboxes as an empty NumPy array with shape (0, 4)
    batch_idx = np.array([], dtype=int)  # Initialize batch_idx as an empty NumPy array
    cls = np.empty((0, 1), dtype=int)  # Initialize cls as an empty 2D NumPy array with a single column

    with open(path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            data = line.strip().split(',')
            cls = np.vstack([cls, [0]])
            bbox_values = np.array([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
            bboxes = np.vstack([bboxes, bbox_values])  # Stack bbox_values vertically to bboxes array
            batch_idx = np.append(batch_idx, index)  # Append index to batch_idx array

    return {"bboxes":bboxes,"cls":cls,"batch_idx":batch_idx}

def load_image(file_path):
    """
    load image using cv2
    """
    try:
        return np.array(cv2.imread(file_path, cv2.IMREAD_UNCHANGED))
    except:
        print("Error couldn't open the file : ",file_path)

def min_max_norm(img):
    minn = np.min(img)
    maxx = np.max(img)
    return (img - minn) / (maxx - minn)


def is_stereo(path):return path.split("/")[8].count("_") > 0


def make_augmentation(image1,image2,bbox1,bbox2,aug):

    if random.choice([0,1]) == 1:
        class_labels_patch1 = ["erosion"] * len(bbox1)
        class_labels_patch2 = ["erosion"] * len(bbox2)

        horizontal = A.Compose([aug],bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels']))
        transformed_patch1 = horizontal(image=image1,bboxes=bbox1,class_labels=class_labels_patch1)  
        bbox1 = transformed_patch1['bboxes']
        image1 = transformed_patch1["image"]

        transformed_patch2 = horizontal(image=image2,bboxes=bbox2,class_labels=class_labels_patch2)
        bbox2 = transformed_patch2['bboxes']
        image2 = transformed_patch2["image"]

    return image1,image2,bbox1,bbox2


def mixup_image(img1,img2):
    """
    mix two image and get one image as result
    """
    alpha = 0.1
    lam =  np.random.beta(alpha,alpha)
    mixed_image1 = lam * img1 + (1 - lam) * img2
    mixed_image2 = lam * img2 + (1 - lam) * img1
    transformed_image = (mixed_image1 + mixed_image2) / 2

    return transformed_image


def get_min_max_dataset(path):
    # get the min and max of a dataset
    df = pd.read_csv(path)
    le_max = -1
    le_min = sys.maxsize

    for i in range(len(df)):
        row = df.iloc[i]
        patch1 = row["patch1"]
        patch2 = row["patch2"]

        file = patch1
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = image[:,:,:3]
        max_1 = np.max(image)
        min_1 = np.min(image)
        if pd.isna(patch2):
            le_max = max(le_max,max_1)
            le_min = min(min_1,le_min)

        else:
            file2 = patch2
            image_2 = cv2.imread(file2, cv2.IMREAD_UNCHANGED)
            image_2 = image_2[:,:,:3]
            max_2 = np.max(image_2)
            min_2 = np.min(image_2)

            le_max = max(max_1,max_2,le_max)
            le_min = min(min_1,min_2,le_min)

    return le_max,le_min



def save_image_using_label(image_path,label_path,save_path):
    """
    method used for quick check & test
    goal: save an image after drawing the bounding box
    using his label file, which mean we do not make any prediction
    """

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = image[:,:,:3]
    image_max = np.max(image)
    image_min = np.min(image)
    image = ((image - image_min) / (image_max - image_min)) * 255

    label = get_label_info(label_path,index=0) # index is not important here
    label_bboxes = label["bboxes"]

    height, width, _ = image.shape

    for box in label_bboxes:
        x, y, w, h = [int(v * width) for v in box]  # Convert relative coordinates to pixels
        x1, y1 = x - w // 2, y - h // 2  # Calculate top-left corner
        x2, y2 = x + w // 2, y + h // 2  # Calculate bottom-right corner
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle

    cv2.imwrite(save_path, image)



def get_mean_std_dataset(csv_path):
    """
    calculate mean and std
    """
    x = pd.read_csv(csv_path)

    means = []
    variance = []

    for i in range(len(x)):
        row = x.iloc[i]
        img_1 = row["patch1"]
        img_1 = np.array(cv2.imread(img_1, cv2.IMREAD_UNCHANGED))
        means.append(np.mean(img_1[:,:,:3], axis=(0,1)))

    mu_rgb = np.mean(means, axis=0)

    variances = []

    for i in range(len(x)):
        row = x.iloc[i]
        img_1 = row["patch1"]
        img_1 = np.array(cv2.imread(img_1, cv2.IMREAD_UNCHANGED))

        var = np.mean((img_1[:,:,:3] - mu_rgb) ** 2, axis=(0,1))
        variances.append(var)

    std_rgb = np.sqrt(np.mean(variances, axis=0))  # std_rgb.shape == (3,)
    print("MEAN: ",mu_rgb)
    print("STD: ",std_rgb)




def get_georeferenced_pos(path,x,y,patch1_stereo=False):
    """
    method to get the georeferenced position from pixels position
    this method is used after prediction when we found the bbox to convert
    the bbox position from pixel to geo
    we also return some useful information used in the shapefile (optional)
    element returned:
        patch_id: col+"_"+ligne
        base_img_id (the patch come from this image): path_split[9]
        the position of x in the base_img (calculated): x_geo_pix
        the position of y in the base_img (calculated): y_geo_pix
    """

    def calculate(img_mere):
        ## the patch path contain information about his position in the base_img such as the column and row
        ligne = path_split[-1].split("_")[-1]
        col = path_split[-1].split("_")[-2]

        dataset = gdal.Open(img_mere, gdal.GA_ReadOnly)
        gt_img_mere = dataset.GetGeoTransform()

        col_pix = int(col)
        lin_pix = int(ligne)

        x_geo_haut_gauche_patch = gt_img_mere[0] + (col_pix * gt_img_mere[1])
        y_geo_haut_gauche_patch = gt_img_mere[3] + (lin_pix * gt_img_mere[5])

        x_pixel = x
        y_pixel = y

        x_geo_pix = x_geo_haut_gauche_patch + (x_pixel * gt_img_mere[1])
        y_geo_pix = y_geo_haut_gauche_patch + (y_pixel * gt_img_mere[5])

        return  x_geo_pix,y_geo_pix,path_split[9],col+"_"+ligne

    # path represent the path of the patch
    path = path[:path.rfind('.')]
    path_split = path.split("/")
    ident = path_split[9]

    #ident represent the base image of the patch
    extension = ".PNG"
    patch_1_2 = ident
    if ident.count("_") > 0 and patch1_stereo:
        extension = "_res.PNG"
    elif ident.count("_") > 0 and patch1_stereo == False: 
        patch_1_2 = ident.split("_")[1] + "_" + ident.split("_")[0]
    
    ident = BASE_IMG_FILE_PATH + ident + "/pair/" + patch_1_2 + extension

    return calculate(ident)


def save_image_with_bbox(img,save_path,bbox,second_head_bbox=None):
    """
    If second head bbox is not None, the result image will contain bounding box from 2 sources (bbox,second_head_bbox)
    """
    for box in bbox.xyxy:
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0, 255, 0), 2)

    if second_head_bbox != None:
        for box in second_head_bbox.xyxy:
            cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0, 255, 0), 2)

    cv2.imwrite(save_path,img)

## YOLOV8 METHOD
def parse_my_detection_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    """
    Code imported
    file: ultralytics/nn/task.py
    Just the line for i, (f, n, m, args) in enumerate(d['backbone'] + d['head])
    has been changed to  for i, (f, n, m, args) in enumerate(d['backbone'] + d['head1'] + d['head2'] )
    """
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out



    for i, (f, n, m, args) in enumerate((d['backbone'] + d['head1']) + d['head2']):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)


    return nn.Sequential(*layers), sorted(save)
