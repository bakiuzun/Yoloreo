import os
import csv
from utils import image_to_label_path

BASE_IMAGE_FILE_PATH = '/share/projects/cicero/objdet/dataset/CICERO_stereo/images/1_Varengeville_sur_Mer'
BASE_LABEL_FILE_PATH = "/share/projects/cicero/objdet/dataset/CICERO_stereo/train_label/1_Varengeville_sur_Mer/"
def write_to_csv(mode,dir_name,ratio_without_label=0.1):

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





mono_images_train = { '201401131052058': 78, '201701261109474': 11, '202304241128373': 57,
                      '201906221113226': 37, '201905141113275': 88, '201707071112468': 72,
                      '201308191121316': 21, '202110141114204': 67}
mono_images_val = {'202106131109358': 54}
mono_images_test  = {'202208271125349': 74}


stereo_images_train =  {'202107221109551_202107221110373': 173, '202109031128506_202109051114435': 171,
                        '202107201124530_202107201125485': 127, '202207081109428_202207081110256': 143,
                        '201809031109535_201809031110026': 31, '201706181108449_201706181110005': 145,
                        '201510041102206_201510041102459': 147, '201802181125339_201802271106473': 205,
                        '202208071128165_202208081121324': 141, '202304181125325_202304181124338': 118,
                        '201610051127479_201610051129175': 133, '202207191124466_202207191125099': 96}

stereo_images_val= {'202106051121186_202106051121491': 144,'201706101120101_201706101121254': 118,
                    '202206041120355_202206041121318': 118,'202105301116513_202105301117321': 155}

stereo_images_test = {'201802171130571_201802171132051': 113,'201510041102114_201510041102549': 98}

#write_to_csv("image_train",'202107221109551_202107221110373')
for i in stereo_images_train:write_to_csv("image_train",i)
for i in stereo_images_val:write_to_csv("image_validation",i)
for i in stereo_images_test:write_to_csv("image_test",i)

for i in mono_images_train:write_to_csv("image_train",i)
for i in mono_images_val:write_to_csv("image_validation",i)
for i in mono_images_test:write_to_csv("image_test",i)
