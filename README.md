# Yoloreo
This project aims to detect erosion on a cliff using stereo and mono images with a modificaiton to the current Yolov8 architecture


## <div align="center"> Dataset </div>
<details open>
<summary>Dataset Information</summary>
The base images were georeferenced (TIFF), we used them to  create patche of images (size: 640x640)

Then for each base image we created a folder and stored the corresponding patch in it.


e.g

    base_img_id_1/
                  patch_1.png
                  patch_2.png
                  patch_3.png

    base_img_id_2/
                  patch_1.png
                  patch_2.png
                  patch_3.png


</details>
NOTE: The format of the dataset is nor coco format nor yolo format.

The images of the dataset is stored into a csv file with 2 column (patch1,patch2).
If patch2 is empty the patch1 will represent a mono image, however if it is not patch1 will represent the first image of the stereo pair, patch2 the second one.

The dataset will return a tensor of images of shape  (batch_size,2,channel,height,width)
the 2 represent -> [images for the first head,images for the second head].
If the there is a mono image we copy the mono image to give it to both head.
The annotation for each image is done after  (check dataset.py) where we use the image files to retrieve labels files and corresponding bounding box.

## <div align="center"> Architecture </div>
We just added another head to the current yolov8 architecture.check the yolov8.yaml file to see the architecture of the model.
With this new architecture we changed the model forward method by adding a cross attention operation after the backbone.

check model.py

## <div align="center"> Training,Validation,Prediction </div>
To train the model:

    python -u main.py

You'll have to change in the main the file of the pre-trained model. For our case we used pre-trained yolov8 medium on coco dataset (you can find it in the folder imported)
