import os
from typing import Dict, Optional

import torch


class SaveCheckPoint:
    def __init__(
        self,
        model: torch.nn.Module,
        log_dir: str,
        monitor: Optional[str] = "val_loss",
        best_mode: str = "min",
        skip_epochs=0,
    ):
        """Init save checkpoint callback.
        Args:
            model (torch.nn.Module): model which weights to save
            logs_dir (str): folder where to save checkpoints
            monitor (Optional[str]): metric to monitor for saving best model
            best_mode (str - "min" or "max"): rule to distinguish 'best' value
            skip_epochs (int): skip saving for number of initials epochs
        """
        self.model = model
        self.log_dir = log_dir
        self.monitor_metric = monitor
        assert best_mode == "min" or best_mode == "max"
        self.best_mode_is_min = best_mode == "min"
        self.skip_epochs = skip_epochs
        self.best_metric_value = None
        if monitor:
            self.filename = "checkpoint_epoch{epoch}_{metric}{value:.3f}.pt"
        else:
            self.filename = "checkpoint_epoch{epoch}.pt"
        self.prev_filepath = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if epoch <= self.skip_epochs:
            return
        if not self.monitor_metric:
            filename = self.filename.format(epoch=epoch)
        else:
            assert logs is not None
            value = logs[self.monitor_metric]
            if self.best_metric_value is not None and (
                self.best_mode_is_min
                and value >= self.best_metric_value
                or not self.best_mode_is_min
                and value <= self.best_metric_value
            ):
                return
            self.best_metric_value = value
            filename = self.filename.format(
                epoch=epoch, metric=self.monitor_metric, value=value
            )
        filepath = os.path.join(self.log_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        print(f"saving checkpoint to {filepath}")
        if self.prev_filepath:
            os.remove(self.prev_filepath)
        self.prev_filepath = filepath
