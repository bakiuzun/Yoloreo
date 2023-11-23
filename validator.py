from ultralytics.models.yolo.detect.val import DetectionValidator

import torch
from ultralytics.utils.ops import Profile
import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.torch_utils import de_parallel

class MyDetectionValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None,dataset=None):

        ## if head 1 True: DO validation using only the output of the first head,
        ## if head 1 False: DO validation using only the output of the second head,
        super().__init__(dataloader, save_dir, args)

        self.dataset = dataset

    @smart_inference_mode()
    def __call__(self, trainer=None):

        #augment = self.args.augment and (not self.training)

        self.device = trainer.device
        self.args.half = self.device.type != 'cpu'  # force FP16 val during training
        #model = trainer.ema.ema or trainer.model
        model = trainer.model

        #model = model.half() if self.args.half else model.float()
        #self.loss_head_1 = torch.zeros_like(trainer.loss_items, device=trainer.device)
        #self.loss_head_2 = torch.zeros_like(trainer.loss_items, device=trainer.device)
        # 3, cls,bboxes, dl loss (Detection Focal Loss)

        self.loss_head_1 = torch.zeros(3).to(device=trainer.device)
        self.loss_head_2 = torch.zeros(3).to(device=trainer.device)

        model.eval()


        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.batch_i = batch_i

            patch_1_annotation,patch_2_annotation = self.dataset.retrieve_annotation(batch,self.device)

            # Preprocess
            #batch = self.preprocess(batch)

            # Inference
            features = model(batch['img'].to(trainer.device))
            preds_head_1 = features["x_1"]
            #preds_head_2 = features["x_2"]

            self.loss_head_1 += trainer.criterion_head_1(preds_head_1,patch_1_annotation)[1]
            #self.loss_head_2 += trainer.criterion_head_2(preds_head_2,patch_2_annotation)[1]

            preds_head_1 = self.postprocess(preds_head_1)
            #preds_head_2 = self.postprocess(preds_head_2)


            img = batch["img"]
            batch_head_1 = batch
            batch_head_1["img"] = img[:,0]
            batch_head_1.update(patch_1_annotation)
            self.update_metrics(preds_head_1, batch_head_1)

            """
            batch_head_2 = batch
            batch_head_2["img"] = img[:,1]
            batch_head_2.update(patch_2_annotation)
            self.update_metrics(preds_head_2, batch_head_2)
            """

        stats = self.get_stats()
        self.check_stats(stats)
        #self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        model.float()
        #results = {**stats, **trainer.label_loss_items( ( (self.loss_head_1.cpu() + self.loss_head_2.cpu()) / 2) / len(self.dataloader), prefix='val')}
        results = {**stats, **trainer.label_loss_items(self.loss_head_1.cpu() / len(self.dataloader), prefix='val')}
        x = {k: round(float(v), 5) for k, v in results.items()}
        print(x)
        return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats


    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""

        self.class_map = list(range(1000))
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = []




    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            cls = cls.unsqueeze(1)
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            #shape = (batch['ori_shape'][si]) # 640,640
            shape = (640,640)
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init

            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()

            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,ratio_pad=None,padding=False)  # native-space pred

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]

                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,ratio_pad=None,padding=False)  # native-space labels

                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)


    # preprocess is already done in the Dataset class
    def preprocess(self, batch):
        return batch
