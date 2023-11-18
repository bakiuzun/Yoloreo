from ultralytics.models.yolo.detect.val import DetectionValidator


class MyDetectionValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)


    # preprocess is already done in the Dataset class
    def preprocess(self, batch):
        return batch
