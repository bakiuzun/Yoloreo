"""
author Uzun Baki
-- u.officialdeveloper@gmail.com
"""

from ultralytics.cfg import get_cfg
import torch
from torch.cuda import amp
import math
from torch.utils.data import DataLoader
import numpy as np
from torch import optim
from torch.cuda import amp
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import (EarlyStopping, one_cycle,de_parallel)
import warnings
from ultralytics.utils.checks import check_amp
from ultralytics.models.yolo.detect import DetectionPredictor
import warnings
from copy import deepcopy
import shutil
from pathlib import Path
import copy
from dataset import CliffDataset
from validator import YoloreoValidator
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK, TQDM, __version__, callbacks, clean_url, colorstr, emojis)
from datetime import datetime
from ultralytics.engine.trainer import BaseTrainer


class YoloreoTrainer(BaseTrainer):
    def __init__(self,cfg,train_path,valid_path,model,overrides=None):
        """
        Detection Trainer used to train,validate and save the metrics
        call the method train ti train and validate the model
        """
        self.args = get_cfg(cfg, overrides)

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_save_dirs()

        ## class names and number of class should be attached too,  check main.py
        self.model = model
        self.model.to(self.device)
        self.model.args = self.args


        self.train_dataset = CliffDataset(path=train_path)
        self.validation_dataset = CliffDataset(mode=valid_path)

        self.validator = None # class Validator

        self.metrics_head_1 = None
        self.metrics_head_2 = None

        ## criterion init
        self.criterion_head_1 = v8DetectionLoss(de_parallel(model))
        self.criterion_head_2 = v8DetectionLoss(de_parallel(model))

        self.loss_head_1 = None
        self.loss_head_2 = None

        self.loss_items_head_1 = None
        self.loss_items_head_2 = None

        self.lf = None
        self.scheduler = None

        self.start_epoch = 0
        self.epochs = self.args.epochs

        # Epoch level metrics
        self.best_fitness_head1 = None
        self.best_fitness_head2 = None

        self.fitness_head1 = None
        self.fitness_head2 = None

        self.batch_size = self.args.batch

        self.tloss = None
        self.tloss_head_2 = None
        self.tloss_head_1 = None

        self.loss_names = ['Loss']


    def init_save_dirs(self):
        """
        store the weights, configuration file -> cfg.yaml
        and the model architecture yolov8.yaml
        you should include ->  save_dir: your_path in the cfg.yaml
        """
        self.save_dir = Path(self.args.save_dir)
        self.wdir = self.save_dir / 'weights_0'  # weights dir

        counter = 0
        while self.wdir.exists():
            self.wdir = self.save_dir / f'weights_{counter}'  # Modify directory name
            counter += 1

        self.wdir.mkdir()

        cfg_source = Path("cfg.yaml")
        shutil.copy(cfg_source, self.wdir / "cfg.yaml") # change cfg.yaml to your conf name

        model_arch = Path("yolov8.yaml")
        shutil.copy(model_arch, self.wdir / "yolov8.yaml") # change yolov8.yaml to your yolo architecture name

        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths

        current_date = datetime.now().strftime("%d_%H-%M-%S")
        self.csv = self.wdir / "result.csv"

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""

        world_size = 1 if  self.device == "cuda" else 0
        self._setup_train(world_size)
        self._do_train(world_size)

    def _setup_train(self, world_size):
        """
        YOLO official method
        Builds dataloaders and optimizer on correct rank process.
        """

        self.model = self.model.to(self.device)

          # Freeze layers
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                            'See ultralytics.engine.trainer for customization of frozen layers.')
                v.requires_grad = True


        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        self.amp = torch.tensor(check_amp(self.model), device=self.device)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)

        # Dataloaders
        batch_size = self.batch_size

        self.train_loader =  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.args.workers)
        self.validation_loader =  DataLoader(self.validation_dataset ,batch_size=32,shuffle=True)


        ## Validation
        self.init_validator()
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
        self.metrics_head_1 = dict(zip(metric_keys, [0] * len(metric_keys)))
        self.metrics_head_2 = dict(zip(metric_keys, [0] * len(metric_keys)))

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay

        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs

        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay,
                                              iterations=iterations)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        #self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move


    def _do_train(self,world_size):
        """
        Training process + validation for each epoch
        """
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1

        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.model.train()

            pbar = TQDM(enumerate(self.train_loader), total=nb)

            self.tloss = None
            self.tloss_head_2 = None
            self.tloss_head_1 = None

            self.optimizer.zero_grad()
            for i, batch in pbar:

                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                # amp.autocast used to perform operation with float16
                with torch.cuda.amp.autocast(self.amp):

                    batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

                    features = self.model(batch["img"].to(self.device))

                    # we get the labels here by using the img files stored in the batch
                    patch_1_annotation,patch_2_annotation = self.train_dataset.retrieve_annotation(batch,self.device)

                    ## only used to print the cls for the current batch
                    batch['cls'] = torch.cat((patch_1_annotation['cls'], patch_2_annotation['cls']))

                    # x_1 refer to the output from the first head,x_2 from the second head
                    self.loss_head_1, self.loss_items_head_1 = self.criterion_head_1(features["x_1"],patch_1_annotation)
                    self.loss_head_2, self.loss_items_head_2 = self.criterion_head_2(features["x_2"],patch_2_annotation)


                    self.tloss_head_1 = (self.tloss_head_1 * i + self.loss_items_head_1) / (i + 1) if self.tloss_head_1 is not None \
                        else self.loss_items_head_1

                    self.tloss_head_2 = (self.tloss_head_2 * i + self.loss_items_head_2) / (i + 1) if self.tloss_head_2 is not None \
                        else self.loss_items_head_2

                    self.tloss =  (self.tloss_head_1 + self.tloss_head_2)  / 2


                # Backward
                self.scaler.scale((self.loss_head_1 + self.loss_head_2) / 2 ).backward()

                # the optimisation is done after some accumulation, this come from Ultralytics
                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)

                pbar.set_description(
                       ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            # validation step
            self.metrics_head_1,self.fitness_head1, self.metrics_head_2,self.fitness_head2 = self.validate()


            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics_head_1, **self.lr})
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics_head_2, **self.lr})
            self.save_model()


            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

        torch.cuda.empty_cache()



    def optimizer_step(self):
        """
        YOLO method
        Perform a single step of the training optimizer with gradient clipping and EMA update.
        """
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()




    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        YOLO method
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def init_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        self.validator = MyDetectionValidator(dataloader=self.validation_loader,dataset=self.validation_dataset)

    def validate(self):

        metrics_head_1,metrics_head_2 =  self.validator(trainer=self)

        fitness_head_1 = metrics_head_1.pop('fitness')
        fitness_head_2 = metrics_head_2.pop('fitness')
        #fitness_both = metrics_both.pop('fitness')

        print("FITNESS 1",fitness_head_1)
        print("FITNESS 2",fitness_head_2)
        #print("FITNESS BOTH ",fitness_both)
        print("BEST FITNESS 1 ",self.best_fitness_head1)
        print("BEST FITNESS 2 ",self.best_fitness_head2)
        #print("BEST FITNESS BOTH",self.best_fitness_both)

        # for the first epoch fitness will be None at start
        if self.best_fitness_head1 == None:
            self.best_fitness_head1 = fitness_head_1
            self.best_fitness_head2 = fitness_head_2
            #self.best_fitness_both  = fitness_both
        else:
            ## you can change the way you store the best_fitness by updating each head separatly
            if fitness_head_1 > self.best_fitness_head1 and fitness_head_2 > self.best_fitness_head2:
                self.best_fitness_head1 = fitness_head_1
                self.best_fitness_head2 = fitness_head_2
                #self.best_fitness_both  = fitness_both


        return metrics_head_1,fitness_head_1,metrics_head_2,fitness_head_2 #, metrics_both


    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import pandas as pd  # scope for faster startup

        ckpt = {
            'epoch': self.epoch,
            'model': deepcopy(de_parallel(self.model)).half(),
            'date': datetime.now().isoformat()
            }

        """
        save only if the 2 head surpass their precedent best fitness
         you can change by saving 2 model (one where the head1 perform his best,second where the  head2 perform his best)
         here one of the head may surpass his best fitness and the best of the second head however if the second head do not surpass
         his actual best, the model won't be saved.
        """
        if self.best_fitness_head1 == self.fitness_head1 and self.best_fitness_head2  == self.fitness_head2:
            torch.save(ckpt, self.best)
