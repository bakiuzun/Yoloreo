# Yoloreo
## This project aims to detect erosion on a cliff using stereo and mono images with a modificaiton to the current Yolov8 architecture

## Architecture
check the yolov8.yaml file to see the architecture of the model. We just added another head to the current yolov8 medium architecture.
in the model.py file you can see how does our model work. The images come with a shape (batch_size,2,channel,height,width).
Explanation of the 2 in the image shape. When building the dataset (check dataset.py) we should handle mono and stereo images as the dataset contain both.
