#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet with pytorch lightning.
Modified from provided train.py example
"""

import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

import pytorch_lightning as pl
# from krotorch.utils import get_cosine_schedule_with_warmup
# from krotorch.vitam.data import show_tensor
import timm


    
class LitTimm(pl.LightningModule):
    """A simple wrapper around all the components of timm's train.py for classification"""
    def __init__(
        self,
        model: str, # Name of the TIMM model architecture to load
        lr=1.5e-4,  # Base learning rate, taken from MAE paper for pretraining
        weight_decay=0.05,  # Weight decay
        betas=(0.9, 0.99),  # Parameters for weight decay optimizer
        warmup_epochs=1,  # Num warmup epochs in lr_scheduler
        lr_sched_max_epochs=600,  # Max number of training epochs for scheduler
        check_every=500,  # Perform more extensive checks every `check_every` steps
        train_betas=False,
    ):
        super().__init__()
        self.decode_only_occluded = decode_only_occluded
        self.model = timm.create_model(model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs
        self.lr_sched_max_epochs = lr_sched_max_epochs
        self.check_every = check_every

        self._metrics_logged = False
        self.save_hyperparameters()

    def configure_optimizers(self):
        # Might need to consider alphas here
        params = [v for k, v in self.vit.named_parameters() if "betas" not in k]
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.warmup_epochs, self.lr_sched_max_epochs
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=lr_scheduler,
                interval="epoch",
                frequency=1,
            ),
        )

    def forward(self, x):
        y = self.model(x, False)
        return y
    
    def loss(self, y, yhat):
        pass

    def _decode_batch(
        self, batch: PatchAndMaskDatasetOutput
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # Returns (y, yhat, mask) == (BNCHW, BNCHW, BN), where yhat has been corrupted and recreated by ViT
        """Simple helper function for the loss in training and val steps"""
        img_tokens, masked_img_tokens, mask = [
            batch[k] for k in ["img_tokens", "masked_img_tokens", "mask"]
        ]
        decoded_patches = self.vit(masked_img_tokens, mask)
        return img_tokens, decoded_patches, mask

    def _show_img_grid(
        self,
        imgs: List[torch.Tensor], # List of images to show in each cell, from left to right
        nv: int = 36,  # Max number of images to show of provided batch
    ) -> torch.Tensor:  # CHW Image representating grid of (masked,yhat,y) cells
        """Show qualitative reconstructions of a batch of images"""
        imgs = [
            show_tensor(self.vit.patcher.unpatchify(x[:nv]))
            for x in imgs
        ]
        nrow = int(np.ceil(np.sqrt(len(imgs[0]))))
        imgs = [x.clamp(0,1) for x in imgs]
        # if yhat_partial is not None:
        #     imgs = [occluded, yhat.clamp(0, 1), yhat_partial.clamp(0, 1), y.clamp(0, 1)]
        # else:
        #     imgs = [occluded, yhat.clamp(0, 1), y.clamp(0, 1)]

        plot_img = make_grid(
            rearrange(imgs, "s b c h w -> b c h (s w)"),
            nrow=nrow,
            normalize=False,
        )
        return plot_img

    def training_step(self, batch, batch_idx):
        tb = self.logger.experiment
        if not self._metrics_logged:
            tb.add_text(f"Model Summary", self.model_summary(), self.global_step)
            self._metrics_logged = True

        # Decode and calc loss
        y, masked_img_tokens, mask = [
            batch[k] for k in ["img_tokens", "masked_img_tokens", "mask"]
        ]
        yhat = self.vit(masked_img_tokens, mask)
        loss = self.loss(y, yhat, mask)

        with torch.no_grad():
            # Collect additional stats on train batch?
            if (self.global_step) % self.check_every == 0:
                tb.add_scalar(f"train.loss", loss, self.global_step)

            # Reconstructions
            if (batch_idx % 1000) == 0:
                yhat_keep_open = yhat.clone()
                yhat_keep_open[mask == 0] = y[mask == 0]
                yhat_just_closed = yhat.clone()
                yhat_just_closed[mask == 0] = 0.
                plot_img = self._show_img_grid(
                    [masked_img_tokens,yhat_just_closed, yhat, yhat_keep_open, y], nv=16
                )
                tb.add_image("train.reconstructions", plot_img, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        # Only collect qualitative info for first validation batch
        if batch_idx == 0:
            tb = self.logger.experiment
            y, masked_img_tokens, mask = [
                batch[k] for k in ["img_tokens", "masked_img_tokens", "mask"]
            ]
            yhat = self.vit(masked_img_tokens, mask)
            loss = self.loss(y, yhat, mask)
            tb.add_scalar(f"val.loss", loss, self.global_step)

            # Reconstructions
            yhat_keep_open = yhat.clone()
            yhat_keep_open[mask == 0] = y[mask == 0]
            yhat_just_closed = yhat.clone()
            yhat_just_closed[mask == 0] = 0.
            plot_img = self._show_img_grid([masked_img_tokens, yhat_just_closed, yhat, yhat_keep_open, y], nv=16)
            tb.add_image("val.reconstructions", plot_img, self.global_step)
