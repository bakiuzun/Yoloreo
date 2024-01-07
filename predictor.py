from ultralytics.models.yolo.detect.predict import DetectionPredictor
from dataset import CliffDataset
from torch.utils.data import DataLoader
import torch

from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils import ops
from ultralytics.engine.results import Results

class YoloreoPredictor(DetectionPredictor):
    def __init__(self,cfg,csv_path,model,conf=0.25):

        super().__init__(cfg=cfg)

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = CliffDataset(path=csv_path)
        self.dataloader = DataLoader(self.dataset,batch_size=32)

        self.args.conf = conf
        self.model = model
        self.model = self.model.float()
        self.model.eval()


    def predict(self):


        for batch in self.dataloader:

            self.preprocess(batch)

            # inference
            self.batch = batch

            out = self.model(batch['img'].to(self.device))
            self.results = self.postprocess(out["x_1"], batch["img"][:,0], batch["img"][:,0])
            #print("LEN RESULTS = ",self.results)
            for i in self.results:
                print(i.boxes)
            #for i in self.results:
            #    print("I = ",i)

            break



    def preprocess(self,batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            #img_path = self.batch[0][i]
            img_path = self.batch["im_files_patch1"][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
