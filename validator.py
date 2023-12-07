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


        self.metrics_head_1 =  DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.metrics_head_2 =  DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.metrics_both =  DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

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
        self.loss_both = torch.zeros_like(trainer.loss_items_head_2, device=trainer.device)

        model.eval()

        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val

        for batch_i, batch in enumerate(bar):

            self.batch_i = batch_i

            patch_1_annotation,patch_2_annotation = self.dataset.retrieve_annotation(batch,self.device)
            batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
            #batch,mono_anot,stereo_anot_1,stereo_anot_2 = self.preprocess(batch,patch_1_annotation,patch_2_annotation)
            # Preprocess


            # Inference
            features = model(batch["img"])
            preds_head_1 = features["x_1"]
            preds_head_2 = features["x_2"]
            preds_both = features["mono_res"]

            self.loss_head_1 += trainer.criterion_head_1(preds_head_1,patch_1_annotation)[1]
            self.loss_head_2 += trainer.criterion_head_2(preds_head_2,patch_2_annotation)[1]
            #self.loss_both += trainer.criterion_head_2(preds_head_2,patch_2_annotation)[1]

            preds_head_1 = self.postprocess(preds_head_1)
            preds_head_2 = self.postprocess(preds_head_2)
            #preds_both = self.postprocess(preds_both)



            patch_1_annotation["img"] = batch["img"][:,0]
            patch_2_annotation["img"] = batch["img"][:,1]
            self.update_metrics(preds_head_1, patch_1_annotation,head_name="head1")
            self.update_metrics(preds_head_2, patch_2_annotation,head_name="head2")
            #self.update_metrics(preds_both, mono_anot,head_name="both")


        stats_head_1 = self.get_stats(head_name="head1")
        stats_head_2 = self.get_stats(head_name="head2")
        #stats_both = self.get_stats(head_name="both")

        self.print_results(head="head1")
        self.print_results(head="head2")
        #self.print_results(head="both")

        model.float()
        #results = {**stats, **trainer.label_loss_items( ( (self.loss_head_1.cpu() + self.loss_head_2.cpu()) / 2) / len(self.dataloader), prefix='val')}
        results_head_1 = {**stats_head_1, **trainer.label_loss_items(self.loss_head_1.cpu() / len(self.dataloader), prefix='val')}
        results_head_2 = {**stats_head_2, **trainer.label_loss_items(self.loss_head_2.cpu() / len(self.dataloader), prefix='val')}
        #results_both = {**stats_both, **trainer.label_loss_items(self.loss_both.cpu() / len(self.dataloader), prefix='val')}

        return [{k: round(float(v), 5) for k, v in results_head_1.items()},{k: round(float(v), 5) for k, v in results_head_2.items()} ]
        #return [{k: round(float(v), 5) for k, v in results_head_1.items()},{k: round(float(v), 5) for k, v in results_head_2.items()},{k: round(float(v), 5) for k, v in results_both.items()} ]  # return results as 5 decimal place floats


    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""


        self.class_map = list(range(1000))
        self.names = model.names
        self.nc = len(model.names)

        self.metrics.names = self.names
        self.metrics_head_1.names = self.names
        self.metrics_head_2.names = self.names
        self.metrics_both.names = self.names

        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)

        self.seen = 0
        self.jdict = []
        self.stats = []
        self.stats_head_1 = []
        self.stats_head_2 = []
        self.stats_both = []




    def get_stats(self,head_name="head1"):
        """Returns metrics statistics and results dictionary."""

        if head_name == "head1":
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats_head_1)]  # to numpy
        elif head_name == "head2":
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats_head_2)]  # to numpy
        elif head_name == "both":
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats_both)]  # to numpy

        if len(stats) and stats[0].any():
            if head_name == "head1":
                self.metrics_head_1.process(*stats)
            elif head_name == "head2":
                self.metrics_head_2.process(*stats)
            elif head_name == "both":
                self.metrics_both.process(*stats)

        self.nt_per_class = np.bincount(stats[-1].astype(int),minlength=self.nc)  # number of targets per class
        if head_name == "head1":
            return self.metrics_head_1.results_dict
        elif head_name == "head2":
            return self.metrics_head_2.results_dict

        return self.metrics_both.results_dict

    def update_metrics(self, preds, batch,head_name="head1"):
        """Metrics."""

        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]

            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            #shape = (batch['ori_shape'][si]) # 640,640
            shape = (640,640)
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init

            self.seen += 1

            if npr == 0:
                if nl:
                    if head_name == "head1":
                        self.stats_head_1.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))

                    elif head_name == "head2":
                        self.stats_head_2.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))

                    elif head_name == "both":
                        self.stats_both.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
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


            if head_name == "head1":
                self.stats_head_1.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)
            elif head_name == "head2":
                self.stats_head_2.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            elif head_name == "both":
                self.stats_both.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

    # preprocess is already done in the Dataset class
    def preprocess(self, batch,annotation_1,annotation_2):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

        stereo_indices = []
        mono_indices = []

        for i,is_stereo in enumerate(batch["stereo"]):
            if is_stereo:
                stereo_indices.append(i)
            else:
                mono_indices.append(i)

        stereo_indices = np.array(stereo_indices)
        mono_indices = np.array(mono_indices)


        # [:,0] because the shape is (batch,2,3,600,600) each batch contain 2 image, if it's mono the two images are the same so we just
        # take one representation as the other one is only a copy of the first
        batch["mono_images"] = batch["img"][mono_indices][:,0]
        batch["stereo_images"] =  batch["img"][stereo_indices]

        ## it can be annotation_1 or annotation_2 as when it's mono they both have the same annotation
        mono_anot = {"img":  batch["mono_images"],
                              "bboxes":  annotation_1["bboxes"][mono_indices],
                              "cls":annotation_1["cls"][mono_indices],
                              "batch_idx":annotation_1["batch_idx"][mono_indices]}

        stereo_anot_1 = {"img":  batch["stereo_images"][:,0],
                              "bboxes":  annotation_1["bboxes"][stereo_indices],
                              "cls":annotation_1["cls"][stereo_indices],
                              "batch_idx":annotation_1["batch_idx"][stereo_indices]}

        stereo_anot_2 = {"img":  batch["stereo_images"][:,1],
                              "bboxes":  annotation_2["bboxes"][stereo_indices],
                              "cls":annotation_2["cls"][stereo_indices],
                              "batch_idx":annotation_2["batch_idx"][stereo_indices]}

        return batch,mono_anot,stereo_anot_1,stereo_anot_2






    def evaluate(self,model,device):

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

        if head == "head1":
            pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics_head_1.keys)  # print format
        elif head == "head2":
            pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics_head_2.keys)  # print format

        elif head == "both":
            pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics_both.keys)  # print format



        if head == "head1":
            LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics_head_1.mean_results()))
        elif head == "head2":
            LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics_head_2.mean_results()))

        elif head == "both":
            LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics_both.mean_results()))
