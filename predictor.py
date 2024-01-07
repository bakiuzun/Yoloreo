from ultralytics.models.yolo.detect.predict import DetectionPredictor
from dataset import CliffDataset
from torch.utils.data import DataLoader
import torch

from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils import ops
from ultralytics.engine.results import Results
from utils import save_image_with_bbox

class YoloreoPredictor(DetectionPredictor):
    def __init__(self,cfg,csv_path,model,conf=0.25):

        super().__init__(cfg=cfg)

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = CliffDataset(path=csv_path)

        self.batch_size = 32
        self.dataloader = DataLoader(self.dataset,shuffle=False,batch_size=self.batch_size)

        self.args.conf = conf
        self.model = model
        self.model = self.model.float()
        self.model.eval()


    def predict(self,save_res=False):


        for idx,batch in enumerate(self.dataloader):

            self.preprocess(batch)

            # inference
            self.batch = batch

            out = self.model(batch['img'].to(self.device))
            self.result_head_1 = self.postprocess(out["x_1"], batch["img"][:,0], batch["img"][:,0])
            self.result_head_2 = self.postprocess(out["x_2"], batch["img"][:,1], batch["img"][:,1],head="head2")

            self.handle_result(save_res,idx)


    def handle_result(self,save_img_res,idx):

        for i in range(len(self.result_head_1)):
            res_head_1 = self.result_head_1[i]
            res_head_2 = self.result_head_2[i]
            ## STEREO
            if res_head_1.path != res_head_2.path:
                if save_img_res:
                    save_image_with_bbox(res_head_1.orig_img,f"head1_{i+(self.batch_size+idx)}.png",res_head_1.boxes)
                    save_image_with_bbox(res_head_2.orig_img,f"head2_{i+(self.batch_size*idx)}.png",res_head_2.boxes)
            else:
                ## MONO MERGE RESULT
                ## THE IMAGE IS THE SAME
                if save_img_res:
                    save_image_with_bbox(res_head_1.orig_img,f"head1_head2_{i+(self.batch_size*idx)}.png",res_head_1.boxes,res_head_2.boxes)


    def preprocess(self,batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

    def postprocess(self, preds, img, orig_imgs,head="head1"):


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

            ## tiny update comapred to the official yolov8
            img_path = self.batch["im_files_patch1"][i]
            if head == "head2":
                img_path = self.batch["im_files_patch2"][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
