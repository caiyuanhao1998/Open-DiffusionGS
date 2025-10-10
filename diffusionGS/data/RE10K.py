import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusionGS import register
from diffusionGS.utils.typing import *
from diffusionGS.utils.config import parse_structured

from .base_scene import BaseDataModuleConfig, BaseDataset


@dataclass
class Re10KDataModuleConfig(BaseDataModuleConfig):
    pass

class Re10KDataset(BaseDataset):
    pass


@register("Re10k-datamodule")
class Re10KDataModule(pl.LightningDataModule):
    cfg: Re10KDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(Re10KDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = Re10KDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = Re10KDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = Re10KDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=self.cfg.eval_batch_size, num_workers=self.cfg.num_workers_val)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=self.cfg.eval_batch_size, num_workers=self.cfg.num_workers_val)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=self.cfg.eval_batch_size, num_workers=self.cfg.num_workers_val)