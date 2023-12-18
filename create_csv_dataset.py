import os
import csv
from utils import image_to_label_path
import pandas as pd
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

BASE_IMAGE_FILE_PATH = '/share/projects/cicero/objdet/dataset/CICERO_stereo/images/1_Varengeville_sur_Mer'
BASE_LABEL_FILE_PATH = "/share/projects/cicero/objdet/dataset/CICERO_stereo/train_label/1_Varengeville_sur_Mer/"



def write_to_csv(mode,dir_name,ratio_without_label=0.0):

    csv_file = f"{mode}_split.csv"
    stereo = dir_name.count("_") > 0
    patch_name = "patches_img1"
    patch_dir = os.path.join(BASE_IMAGE_FILE_PATH, dir_name, patch_name)

    txt_files =  [file for file in os.listdir(patch_dir) if file.endswith(".png")]

    nb_total_patch_1 = len(txt_files)
    autorized_nb_images_without_label = round(nb_total_patch_1 * ratio_without_label)

    with open(csv_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for file in txt_files:

            patch_1_file = os.path.join(patch_dir,file)
            label_1_file =  image_to_label_path(patch_1_file,patch1=True)
            this_file_contain_object = label_contain_object(label_1_file)

            if stereo:
                # get the image and label file of the second patch
                second_file_path = get_second_patch_file_path(dir_name,file,patch_dir)
                label_2_file =  image_to_label_path(second_file_path,patch1=False)
                this_second_file_contain_object = label_contain_object(label_2_file)


                # if one of the patch contain an object we save it
                if this_file_contain_object or this_second_file_contain_object:
                    csv_writer.writerow([patch_1_file,second_file_path])
                else:
                    # if no object in 2 patches we check if we can save it
                    if autorized_nb_images_without_label > 0:
                        autorized_nb_images_without_label -= 1
                        csv_writer.writerow([patch_1_file,second_file_path])

            else:
                if this_file_contain_object:
                    csv_writer.writerow([patch_1_file])
                else:
                    if autorized_nb_images_without_label > 0:
                        autorized_nb_images_without_label -= 1
                        csv_writer.writerow([patch_1_file])

def label_contain_object(label_file):
    with open(label_file, 'r') as file:
            # Read lines from the file
        lines = file.readlines()
        return True if len(lines) > 0 else False


def get_second_patch_file_path(dir_name,file,patch_dir):
    second_patch_identifier = dir_name.split("_")[1]
    splitted_file = patch_dir.split("/")

    patch_cm = splitted_file[-1]
    patch_cm = patch_cm.replace("1","2")

    extension = file.split("_")
    extension[1] = second_patch_identifier

    second_file_path =  '/'.join(splitted_file[:-1]) + "/" + patch_cm + "/" + "_".join(extension)
    return second_file_path


def ret_box(path):
    """
    tiny change compared to the method in the utils file
    """
    bboxes = []  # Initialize bboxes as an empty NumPy array with shape (0, 4)

    with open(path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            data = line.strip().split(',')
            bbox_values = [float(data[1]), float(data[2]), float(data[3]), float(data[4]),"erosion"]
            bboxes.append(bbox_values)

    return bboxes


def create_new_image_path(base_image_path,index):
    splitted = base_image_path.split("/")
    last_name = splitted[-1]
    last_name_splitted = last_name.split("_")
    last_name_splitted[0] = last_name_splitted[0] + f"aug{index}"

    splitted[-1] = '_'.join(last_name_splitted)

    return "/".join(splitted)



def get_random_patch_index(df):

    random_index = random.randint(0, len(df) - 1)

    while (pd.isna(df.iloc[random_index]["patch2"]) == True):
        random_index = random.randint(0, len(df) - 1)

    return random_index



def save_img(path,img):

    minn = np.min(img)
    maxx = np.max(img)

    img = ((img - minn) / (maxx - minn)) * 255

    cv2.imwrite(path,img)



def delete_augmented(file,only_stereo=True):
    ## only stereo not used right now
    from utils import (image_to_label_path)

    df = pd.read_csv(file)

    for i in range(len(df)):
        row = df.iloc[i]

        image_path = row["patch1"]

        stereo = pd.isna(row["patch2"]) == False
        if stereo:
            if "aug" in image_path:
                os.remove(image_path)
                os.remove(image_to_label_path(image_path))

                path_2 = row["patch2"]
                os.remove(path_2)
                os.remove(image_to_label_path(path_2,False))


def mixup_image(img1,img2):

    alpha = 0.5
    lam =  np.random.beta(alpha,alpha)
    mixed_image1 = lam * img1 + (1 - lam) * img2
    mixed_image2 = lam * img2 + (1 - lam) * img1
    transformed_image = (mixed_image1 + mixed_image2) / 2

    return transformed_image

def write_annot(file,bboxes,option="w"):

    with open(file,option) as new_annot:
        for boxe in bboxes:
            thing_to_save = f"0,{boxe[0]},{boxe[1]},{boxe[2]},{boxe[3]}\n"
            new_annot.write(thing_to_save)

        new_annot.close()



def augment_and_save(csv_path):
    import albumentations as A
    from utils import (image_to_label_path)
    df = pd.read_csv(csv_path)

    # THE AUGS
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo'))


    for i in range(len(df)):
        row = df.iloc[i]

        image_path = row["patch1"]
        label = image_to_label_path(image_path,True)
        bboxes = ret_box(label)

        stereo = pd.isna(row["patch2"]) == False
        if stereo:
            image = np.array(cv2.imread(image_path,cv2.IMREAD_UNCHANGED))
            image = image[:,:,:3]


            stereo_path = row["patch2"]
            image_2 = np.array(cv2.imread(stereo_path,cv2.IMREAD_UNCHANGED))
            image_2 = image_2[:,:,:3]
            label_2 = image_to_label_path(stereo_path,False)
            bboxes_2 = ret_box(label_2)
            for i in range(5):
                random_index = get_random_patch_index(df)
                random_second_img_path = df.iloc[random_index]["patch2"]
                random_second_img =   np.array(cv2.imread(random_second_img_path,cv2.IMREAD_UNCHANGED))
                random_second_img = random_second_img[:,:,:3]
                label_random_patch = image_to_label_path(random_second_img_path,False)
                bboxes_random = ret_box(label_random_patch)

                mixup_img  =  mixup_image(image_2,random_second_img)


                random_second_img_path_2 = df.iloc[random_index]["patch1"]
                random_second_img_2 =   np.array(cv2.imread(random_second_img_path_2,cv2.IMREAD_UNCHANGED))
                random_second_img_2 = random_second_img_2[:,:,:3]
                label_random_patch2 = image_to_label_path(random_second_img_path_2,True)
                bboxes_random_2 = ret_box(label_random_patch2)

                mixup_img_2  =  mixup_image(image,random_second_img_2)


                new_img_1_path = create_new_image_path(image_path,i)
                new_img_2_path = create_new_image_path(stereo_path,i)


                label1 = image_to_label_path(new_img_1_path,True)
                label2 = image_to_label_path(new_img_2_path,False)

                write_annot(label1,bboxes_random_2,option="a")
                write_annot(label1,bboxes,option="a")
                cv2.imwrite(new_img_1_path,mixup_img_2.astype(np.uint16))
                #cv2.imwrite(new_img_1_path,image.astype(np.uint16))


                write_annot(label2,bboxes_random,option="a")
                write_annot(label2,bboxes_2,option="a")
                cv2.imwrite(new_img_2_path,mixup_img.astype(np.uint16))

            """
            for i in range(10):
                new_image_path =  create_new_image_path(image_path,i)


                label = image_to_label_path(new_image_path,True)

                transformed = transform(image=image,bboxes=bboxes)
                transformed_bboxes = transformed['bboxes']
                transformed_image = transformed["image"]

                with open(label,'w') as new_annot:
                    for boxe in transformed_bboxes:
                        thing_to_save = f"0,{boxe[0]},{boxe[1]},{boxe[2]},{boxe[3]}\n"
                        new_annot.write(thing_to_save)

                    new_annot.close()

                cv2.imwrite(new_image_path,transformed_image)
            """


mono_images_train = {'201401131052058': 78, '201701261109474': 11, '202304241128373': 57,
                      '201906221113226': 37, '201905141113275': 88, '201707071112468': 72, '202110141114204': 67}
mono_images_val = {'202106131109358': 54,'202208271125349': 74}


stereo_images_train =  {'202107221109551_202107221110373': 173, '202109031128506_202109051114435': 171,
                        '202107201124530_202107201125485': 127, '202207081109428_202207081110256': 143,
                        '201809031109535_201809031110026': 31, '201706181108449_201706181110005': 145,
                        '201510041102206_201510041102459': 147, '201802181125339_201802271106473': 205,
                        '202208071128165_202208081121324': 141, '202304181125325_202304181124338': 118,
                        '201610051127479_201610051129175': 133, '202207191124466_202207191125099': 96}

stereo_images_val= {'202106051121186_202106051121491': 144,'201706101120101_201706101121254': 118,
                    '202206041120355_202206041121318': 118,'202105301116513_202105301117321': 155,
                    '201802171130571_201802171132051': 113,'201510041102114_201510041102549': 98}


#augment_and_save('dataset_for_augs.csv')



for i in stereo_images_train:write_to_csv("image_train",i)
#for i in stereo_images_val:write_to_csv("image_valid",i)


for i in mono_images_train:write_to_csv("image_train",i)
#for i in mono_images_val:write_to_csv("image_valid",i)
