from ultralytics.cfg import get_cfg
import torch
from torch.cuda import amp
import math
from torch.utils.data import DataLoader
import numpy as np
from torch import nn, optim
from torch.cuda import amp
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import (EarlyStopping, one_cycle,de_parallel)
import warnings
from torchvision.transforms import ToTensor
from ultralytics.utils.checks import check_amp
from ultralytics.models.yolo.detect import DetectionPredictor
import warnings
from copy import deepcopy
from datetime import datetime


class MyTrainer():
    def __init__(self,cfg,model,dataset,overrides=None):

        self.args = get_cfg(cfg, overrides)

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.validator = None

        self.model = model
        self.model.to(self.device)
        self.model.args = self.args

        self.dataset = dataset
        self.metrics = None

        ## criterion init
        self.criterion_head_1 = v8DetectionLoss(self.model)
        self.criterion_head_2 = v8DetectionLoss(self.model)

        self.lf = None

        self.scheduler = None

        self.start_epoch = 0
        self.epochs = self.args.epochs

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.batch_size = self.args.batch
        self.loss = None
        self.tloss = None
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
        batch_size = self.batch_size // max(world_size, 1)

        self.train_loader =  DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        """
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            #self.ema = ModelEMA(self.model)

        """

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        print("TRAIN_LOADER_DATASET = ",len(self.train_loader.dataset))
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

        predictor = DetectionPredictor()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1

        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):

            self.epoch = epoch
            self.model.train()
            pbar = enumerate(self.train_loader)

            self.tloss = None
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

                    patch_1_annotation,patch_2_annotation = self.dataset.retrieve_annotation(batch,self.device)
                    batch['cls'] = patch_1_annotation['cls']

                    self.loss_head_1, self.loss_items_head_1 = self.criterion_head_1(features["x_1"],patch_1_annotation)
                    #self.loss_head_2, self.loss_items_head_2 = self.criterion_head_2(features["x_2"],patch_2_annotation)

                    self.tloss = (self.tloss * i + self.loss_items_head_1) / (i + 1) if self.tloss is not None \
                        else self.loss_items_head_1

                # Backward
                self.scaler.scale(self.loss_head_1).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)


                print("INFO |",(('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1])))

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()


            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            if self.epoch % 20 == 0:
                self.save_model()


        torch.cuda.empty_cache()
        print("SAVE MODEL ")
        #model_path = 'my_model.pth'

        #torch.save(self.model.state_dict(), model_path)

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        #if self.ema:
        #    self.ema.update(self.model)


    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        #import pandas as pd  # scope for faster startup
        #metrics = {**self.metrics, **{'fitness': self.fitness}}
        #results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}
        """
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat(),
            'version': __version__}
        """
        ckpt = {
            'epoch': self.epoch,
            'model': deepcopy(de_parallel(self.model)).half(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat()}

        # Save last and best
        torch.save(ckpt,f"/share/projects/cicero/checkpoints_baki/lr_{self.args.lrf}_epoch_{self.epoch}.pt")




    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if name == 'auto':
            #nc = getattr(model, 'nc', 10)  # number of classes
            nc = 1

            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
                'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)

        return optimizer
