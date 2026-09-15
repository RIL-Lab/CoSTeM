import torch
import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    """Base class of all embryo video models.

    A concrete model is expected to implement the methods below, which are called
    by ``scripts/train_new_version.py``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def build_dataloader(self, logger=None, world_size=1, global_rank=0):
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(self, logger=None):
        raise NotImplementedError

    @abstractmethod
    def build_scheduler(self, optimizer, n_iter_per_epoch, logger=None):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler,
                        train_loader, config, logger, writter):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate(epoch, model, criterion, val_loader, config, logger, writter):
        raise NotImplementedError
