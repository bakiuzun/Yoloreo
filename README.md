# Yoloreo
This project aims to detect erosion on a cliff using stereo and mono images with a modificaiton to the current Yolov8 architecture


## <div align="center"> Dataset </div>
<details close>
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

<details close>
<summary>Dataset Creation</summary>
The images files are stored into a csv file and we can create multiple type of dataset by changing some parameter check: create_csv_dataset.py
In the file you'll see some important method to create a dataset, augment a dataset and save augmented images with label files.

For example you can choice the percentage of images without labels to be added into the csv file
The identifier in the file correspond to the identifier talked in the "Dataset Information" section.
If you want to augment a dataset that contain at least 1 annotation you should firstly call the method <code>write_to_csv(mode,dir_name,ratio_without_label=0.0) </code>
then call the method <code>augment_and_save(csv_path) </code> this method will use the mixup augmentation for stereo images and some basics augmentation for the mono images
</details>
NOTE: The format of the dataset is neither coco format nor yolo format.

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

By default it will load yolov8 medium weight pre-trained on coco-dataset.As the pre-trained weight is only for one head we copied the weight of the first head to the second one (check model.py).

To change the training settings (nb epoch, ....,) you'll have to modify the cfg.yaml. NOTE: even if there are multiples parameters in the cfg.yaml our code don't take each of them into account for example some parameters of augmentation are given however in our case the augmentation is done before the training process (we saved the augmented images) so modifiying the parameter values won't change the training or validation process.
<details close>
<summary>Modifiable Parameter</summary>
These are the parameters that you can modify and it will be taken in account during training,validation & prediction
for the prediction and evaluation of a model,the conf paramater  should be  specified  directly in the code, check save_pred.py,evaluate.py

    epochs: 250 # (int) number of epochs to train for
    batch: 16 # (int) number of images per batch
    save_dir: "/share/projects/cicero/checkpoints_baki" # path were the weights will be saved
    conf: # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val) you can modify it in my code
    iou: 0.7 # (float) intersection over union (IoU) threshold for NMS
    max_det: 300 # (int) maximum number of detections per image
    freeze: None
    lr0: 0.01 # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    lrf: 0.01 # (float) final learning rate (lr0 * lrf)
    momentum: 0.937 # (float) SGD momentum/Adam beta1
    weight_decay: 0.0001 # (float) optimizer weight decay 5e-4
    warmup_epochs: 3.0 # (float) warmup epochs (fractions ok)
    warmup_momentum: 0.8 # (float) warmup initial momentum
    warmup_bias_lr: 0.1 # (float) warmup initial bias lr
    box: 7.5 # (float) box loss gain
    cls: 0.5 # (float) cls loss gain (scale with pixels)
    dfl: 1.5 # (float) dfl loss gain

</details>
To evaluate a model:

    python -u evaluate.py
you'll be able to change some parameter of confidence and the path of the validation file. check evaluate.py

To predict & save:

    python -u save_pred.py

2 important things about prediction. You can save the predicted image with bbox result and also save a shapefile that will contain the georeferenced bbox coordinates (check predictor.py and utils.py to understand how we transform from pixels position to georeferenced position.
even if you have only 1 image to predict you should create a csv file and put the path of the image in it with column (patch1,patch2), where patch2 can be empty if not stereo


## <div align="center"> Result </div>
We compared the result of our Yoloreo model with the Baseline Model (yolov8) where we don't have the notion of stereo so all images (stereo and mono) are seen as mono images.
After 5 benchmark each constitued of 4 runs (total of 20 runs with different settings) the Yoloreo model surpass the baseline model in each benchmark.
