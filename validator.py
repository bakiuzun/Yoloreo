"""
author Uzun Baki
-- u.officialdeveloper@gmail.com
"""

from ultralytics.models.yolo.detect.val import DetectionValidator

import torch
from ultralytics.utils.ops import Profile
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import de_parallel, smart_inference_mode

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

class YoloreoValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None,dataset=None):

        ## if head 1 True: DO validation using only the output of the first head,
        ## if head 1 False: DO validation using only the output of the second head,
        super().__init__(dataloader, save_dir, args)


        self.metrics_head_1 =  DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.metrics_head_2 =  DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

        self.dataset = dataset


    @smart_inference_mode()
    def __call__(self, trainer=None):

        #augment = self.args.augment and (not self.training)

        self.device = trainer.device
        self.args.half = self.device.type != 'cpu'  # force FP16 val during training

        model = trainer.model
        model = model.float()

        self.loss_head_1 = torch.zeros_like(trainer.loss_items_head_1, device=trainer.device)
        self.loss_head_2 = torch.zeros_like(trainer.loss_items_head_2, device=trainer.device)

        model.eval()

        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val

        for batch_i, batch in enumerate(bar):

            self.batch_i = batch_i

            patch_1_annotation,patch_2_annotation = self.dataset.retrieve_annotation(batch,self.device)
            batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255


            # Inference
            features = model(batch["img"])
            preds_head_1 = features["x_1"]
            preds_head_2 = features["x_2"]


            self.loss_head_1 += trainer.criterion_head_1(preds_head_1,patch_1_annotation)[1]
            self.loss_head_2 += trainer.criterion_head_2(preds_head_2,patch_2_annotation)[1]

            preds_head_1 = self.postprocess(preds_head_1)
            preds_head_2 = self.postprocess(preds_head_2)


            patch_1_annotation["img"] = batch["img"][:,0]
            patch_2_annotation["img"] = batch["img"][:,1]
            self.update_metrics(preds_head_1, patch_1_annotation,head_name="head1")
            self.update_metrics(preds_head_2, patch_2_annotation,head_name="head2")


        stats_head_1 = self.get_stats(head_name="head1")
        stats_head_2 = self.get_stats(head_name="head2")

        self.print_results(head="head1")
        self.print_results(head="head2")

        model.float()

        results_head_1 = {**stats_head_1, **trainer.label_loss_items(self.loss_head_1.cpu() / len(self.dataloader), prefix='val')}
        results_head_2 = {**stats_head_2, **trainer.label_loss_items(self.loss_head_2.cpu() / len(self.dataloader), prefix='val')}
        return [{k: round(float(v), 5) for k, v in results_head_1.items()},{k: round(float(v), 5) for k, v in results_head_2.items()} ]



    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""


        self.class_map = list(range(1000))
        self.names = model.names
        self.nc = len(model.names)

        self.metrics.names = self.names
        self.metrics_head_1.names = self.names
        self.metrics_head_2.names = self.names

        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)

        self.seen = 0
        self.jdict = []
        self.stats = []
        self.stats_head_1 = []
        self.stats_head_2 = []

    def get_stats(self,head_name="head1"):
        """Returns metrics statistics and results dictionary."""

        if head_name == "head1":
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats_head_1)]  # to numpy
        elif head_name == "head2":
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats_head_2)]  # to numpy

        if len(stats) and stats[0].any():
            if head_name == "head1":
                self.metrics_head_1.process(*stats)
            elif head_name == "head2":
                self.metrics_head_2.process(*stats)

        self.nt_per_class = np.bincount(stats[-1].astype(int),minlength=self.nc)  # number of targets per class
        if head_name == "head1":
            return self.metrics_head_1.results_dict
        elif head_name == "head2":
            return self.metrics_head_2.results_dict


    def update_metrics(self, preds, batch,head_name="head1"):
        """Metrics."""

        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]

            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            #shape = (batch['ori_shape'][si]) # 640,640

            """
                change the shape if it is not 640,640
                if you have images of different shape you should change the dataset.py file  to include the ori_shape attribute
            """
            shape = (640,640)
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init

            self.seen += 1

            if npr == 0:
                if nl:
                    if head_name == "head1":
                        self.stats_head_1.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                        if self.args.plots:self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))

                    elif head_name == "head2":
                        self.stats_head_2.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                        if self.args.plots:self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
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
                if self.args.plots:self.confusion_matrix.process_batch(predn, labelsn)


            if head_name == "head1":
                self.stats_head_1.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)
            elif head_name == "head2":
                self.stats_head_2.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

    def evaluate(self,model,device,conf=0.001):
        """
        evaluation of one model on one dataset with a parameter conf
        this is not used in training
        """
        self.args.conf = conf

        self.device = device
        self.args.half = self.device.type != 'cpu'  # force FP16 val during training

        model = model.float()
        model.eval()

        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val

        for batch_i, batch in enumerate(bar):
            self.batch_i = batch_i

            patch_1_annotation,patch_2_annotation = self.dataset.retrieve_annotation(batch,self.device)

            # Preprocess
            batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

            # Inference
            features = model(batch['img'].to(device))
            preds_head_1 = features["x_1"]
            preds_head_2 = features["x_2"]

            preds_head_1 = self.postprocess(preds_head_1)
            preds_head_2 = self.postprocess(preds_head_2)


            img = batch["img"]
            batch_head_1 = batch
            batch_head_1["img"] = img[:,0]
            batch_head_1.update(patch_1_annotation)
            self.update_metrics(preds_head_1, batch_head_1,head_name="head1")


            batch_head_2 = batch
            batch_head_2["img"] = img[:,1]
            batch_head_2.update(patch_2_annotation)
            self.update_metrics(preds_head_2, batch_head_2,head_name="head2")


        self.get_stats(head_name="head1")
        self.get_stats(head_name="head2")


        self.print_results(head="head1")
        self.print_results(head="head2")
        model.float()


    def print_results(self,head="head1"):
        """Prints training/validation set metrics per class."""
        LOGGER.info(f"HEAD {head}")

        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics_head_1.keys)  # print format

        if head == "head1":
            LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics_head_1.mean_results()))
        elif head == "head2":
            LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics_head_2.mean_results()))


        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir,
                                           names=self.names.values(),
                                           normalize=normalize,
                                           on_plot=self.on_plot)
