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
from pathlib import Path
import copy
from dataset import CliffDataset
from validator import MyDetectionValidator
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK, TQDM, __version__, callbacks, clean_url, colorstr, emojis)
from datetime import datetime
from ultralytics.engine.trainer import BaseTrainer


class MyDetectionTrainer(BaseTrainer):
    def __init__(self,cfg,model,overrides=None):

        self.args = get_cfg(cfg, overrides)

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.save_dir = Path("/share/projects/cicero/checkpoints_baki/")


        ## class names and number of class should be attached too,  check main.py
        self.model = model
        self.model.to(self.device)
        self.model.args = self.args

        self.train_dataset = CliffDataset(mode="train")

        self.validation_dataset = CliffDataset(mode="validation")
        self.validator = None # class Validator

        self.metrics_head_1 = None
        self.metrics_head_2 = None


        self.wdir = self.save_dir / 'weights'  # weights dir
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.csv = self.save_dir / 'results.csv'

        ## criterion init
        self.criterion_head_1 = v8DetectionLoss(self.model)
        self.criterion_head_2 = v8DetectionLoss(self.model)

        self.loss_head_1 = None
        self.loss_head_2 = None

        self.lf = None

        self.scheduler = None

        self.start_epoch = 0
        self.epochs = self.args.epochs

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.batch_size = self.args.batch

        self.tloss = None
        self.tloss_head_2 = None
        self.tloss_head_1 = None


        self.loss_names = ['Loss']

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""

        world_size = 1 if  self.device == "cuda" else 0
        self._setup_train(world_size)
        self._do_train(world_size)

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        self.model = self.model.to(self.device)

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        self.amp = torch.tensor(check_amp(self.model), device=self.device)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)

        # Dataloaders
        batch_size = self.batch_size

        self.train_loader =  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.args.workers)
        self.validation_loader =  DataLoader(self.validation_dataset ,batch_size=batch_size * 2,shuffle=False, num_workers=self.args.workers)


        ## Validation
        self.init_validator()
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
        self.metrics_head_1 = dict(zip(metric_keys, [0] * len(metric_keys)))
        self.metrics_head_2 = dict(zip(metric_keys, [0] * len(metric_keys)))
        #self.ema = ModelEMA(self.model)

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

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1

        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):

            self.epoch = epoch
            self.model.train()
            #pbar = enumerate(self.train_loader)
            pbar = TQDM(enumerate(self.train_loader), total=nb)

            self.tloss = None
            self.tloss_head_2 = None
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
                with torch.cuda.amp.autocast(self.amp):
                    features = self.model(batch["img"].to(self.device))

                    patch_1_annotation,patch_2_annotation = self.train_dataset.retrieve_annotation(batch,self.device)

                    batch['cls'] = torch.cat((patch_1_annotation['cls'], patch_2_annotation['cls']))

                    self.loss_head_1, self.loss_items_head_1 = self.criterion_head_1(features["x_1"],patch_1_annotation)
                    self.loss_head_2, self.loss_items_head_2 = self.criterion_head_2(features["x_2"],patch_2_annotation)


                    self.tloss_head_1 = (self.tloss_head_1 * i + self.loss_items_head_1) / (i + 1) if self.tloss_head_1 is not None \
                        else self.loss_items_head_1

                    self.tloss_head_2 = (self.tloss_head_2 * i + self.loss_items_head_2) / (i + 1) if self.tloss_head_2 is not None \
                        else self.loss_items_head_2

                    self.tloss =  (self.tloss_head_1 + self.tloss_head_2)  / 2


                # Backward
                #self.scaler.scale(self.loss_head_1).backward()
                self.scaler.scale((self.loss_head_1 + self.loss_head_2) / 2 ).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)

                #pbar.set_description(
                 #       ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                  #      (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))

                pbar.set_description(
                       ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                #loss_len_2 = self.tloss_head_2.shape[0] if len(self.tloss_head_2.size()) else 1
                #losses_2 = self.tloss_head_2 if loss_len_2 > 1 else torch.unsqueeze(self.tloss_head_2, 0)


                #print("INFO |",(('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        #(f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1])))

                #print("HEAD 2 INFO |",(('%11s' * 2 + '%11.4g' * (2 + loss_len_2)) %
                        #(f'{epoch + 1}/{self.epochs}', mem, *losses_2, batch['cls_2'].shape[0], batch['img'].shape[-1])))

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()


            self.metrics, self.fitness = self.validate()
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.save_model()

            #self.save_model()
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors


            if self.epoch % 20 == 0:
                self.save_model()


        torch.cuda.empty_cache()
        self.save_model()


    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        #if self.ema:
        #    self.ema.update(self.model)



    def label_loss_items(self, loss_items=None, prefix='train'):
        """
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

        metrics =  self.validator(trainer=self)
        fitness = metrics.pop('fitness', -((self.loss_head_1.detach().cpu().numpy() + self.loss_head_2.detach().cpu().numpy()) / 2 ))  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness



    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import pandas as pd  # scope for faster startup
        metrics = {**self.metrics, **{'fitness': self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            #'ema': deepcopy(self.ema.ema).half(),
            #'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat()
            }

        # Save last and best
        #torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)

        if self.epoch + 1 == self.args.epochs:
            torch.save(ckpt, self.last)


        """
        ckpt = {
            'epoch': self.epoch,
            'model': deepcopy(de_parallel(self.model)).half(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat()

        }
        """

        # Save last and best
        #torch.save(ckpt,f"/share/projects/cicero/checkpoints_baki/cross_lr_{self.args.lrf}_epoch_{self.epoch}.pt")
