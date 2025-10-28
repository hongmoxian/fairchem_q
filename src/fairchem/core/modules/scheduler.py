from __future__ import annotations

import inspect
import math

import torch.optim.lr_scheduler as lr_scheduler

from fairchem.core.common.utils import warmup_lr_lambda
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CosineLRLambda:
    def __init__(
        self,
        warmup_epochs: int,
        warmup_factor: float,
        epochs: int,
        lr_min_factor: float,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.lr_warmup_factor = warmup_factor
        self.max_epochs = epochs
        self.lr_min_factor = lr_min_factor

    def __call__(self, current_step: int) -> float:
        # `warmup_epochs` is already multiplied with the num of iterations
        if current_step <= self.warmup_epochs:
            alpha = current_step / float(self.warmup_epochs)
            return self.lr_warmup_factor * (1.0 - alpha) + alpha
        else:
            if current_step >= self.max_epochs:
                return self.lr_min_factor
            return self.lr_min_factor + 0.5 * (1 - self.lr_min_factor) * (
                1 + math.cos(math.pi * (current_step / self.max_epochs))
            )


class LRScheduler:
    """
    Learning rate scheduler class for torch.optim learning rate schedulers

    Notes:
        If no learning rate scheduler is specified in the config the default
        scheduler is warmup_lr_lambda (fairchem.core.common.utils) not no scheduler,
        this is for backward-compatibility reasons. To run without a lr scheduler
        specify scheduler: "Null" in the optim section of the config.

    Args:
        optimizer (obj): torch optim object
        config (dict): Optim dict from the input config
    """

    def __init__(self, optimizer, config) -> None:
        self.optimizer = optimizer
        self.config = config.copy()
        if "scheduler" in self.config:
            self.scheduler_type = self.config["scheduler"]
        else:
            self.scheduler_type = "LambdaLR"

            def scheduler_lambda_fn(x):
                return warmup_lr_lambda(x, self.config)

            self.config["lr_lambda"] = scheduler_lambda_fn
        if self.scheduler_type != "Null":
            if self.scheduler_type == "ExponentialLR":
                self.scheduler = lr_scheduler.ExponentialLR
                scheduler_args = self.filter_kwargs(config)
                if "final_lr" in self.config and "total_steps" in self.config:
                    initial_lr = self.optimizer.param_groups[0]['lr']
                    final_lr = self.config["final_lr"]
                    total_steps = self.config["total_steps"]
                    gamma = self.calculate_gamma(initial_lr, final_lr, total_steps)
                    scheduler_args["gamma"] = gamma

            elif self.scheduler_type == "ReduceLROnPlateau":
                self.scheduler = ReduceLROnPlateau
                scheduler_args = self.filter_kwargs(config)
                if "factor" in self.config:
                    scheduler_args["factor"] = self.config["factor"]
                if "patience" in self.config:
                    scheduler_args["patience"] = self.config["patience"]
                scheduler_args["verbose"] = True
                scheduler_args["threshold"] = 1e-2
                if "min_lr" in self.config:
                    scheduler_args["min_lr"] = self.config["min_lr"]

            else:
                self.scheduler = getattr(lr_scheduler, self.scheduler_type)
                scheduler_args = self.filter_kwargs(config)
            
            self.scheduler = self.scheduler(optimizer, **scheduler_args)
        
        # Store final_lr if provided
        self.final_lr = self.config.get("final_lr", None)

    def step(self, metrics=None, epoch=None) -> None:
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception("Validation set required for ReduceLROnPlateau.")
            self.scheduler.step(metrics)
        else:
            current_lr = self.get_lr()
            if self.final_lr is not None and current_lr <= self.final_lr:
                return
            self.scheduler.step()

    def filter_kwargs(self, config):
        # adapted from https://stackoverflow.com/questions/26515595/
        sig = inspect.signature(self.scheduler)
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        filter_keys.remove("optimizer")
        return {arg: self.config[arg] for arg in self.config if arg in filter_keys}

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group["lr"]
        return None

    def calculate_gamma(self, initial_lr, final_lr, total_steps):
        return (final_lr / initial_lr) ** (1 / total_steps)