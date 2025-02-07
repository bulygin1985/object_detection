from datetime import datetime
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class TensorboardCallback:
    def __init__(self, log_dir: str = "runs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_log_dir = f"{log_dir}/{timestamp}"
        self.writer = SummaryWriter(unique_log_dir, flush_secs=120)
        self.train_iteration = 0
        self.val_iteration = 0

    def on_train_begin(self, logs: Optional[Dict] = None):
        if logs:
            for key, value in logs.items():
                self.writer.add_text(f"Training/Params/{key}", str(value), 0)

    def on_train_end(self, logs: Optional[Dict] = None):
        self.writer.close()

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        if logs and logs.get("lr"):
            self.writer.add_scalar("Learning Rate", logs["lr"], epoch)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if logs:
            for key, value in logs.items():
                if key in ["train_loss", "val_loss"]:
                    self.writer.add_scalar(f"Epoch/{key}", value, epoch)

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        self.train_iteration += 1
        if logs:
            for key, value in logs.items():
                if key == "lr":
                    self.writer.add_scalar(
                        "Batch/Learning Rate", value, self.train_iteration
                    )

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if logs:
            for key, value in logs.items():
                if key == "loss":
                    self.writer.add_scalar(
                        "Batch/Training Loss", value, self.train_iteration
                    )

    def on_val_begin(self, logs: Optional[Dict] = None):
        self.val_iteration = 0

    def on_val_end(self, logs: Optional[Dict] = None):
        if logs:
            for key, value in logs.items():
                if key == "val_loss":
                    self.writer.add_scalar("Validation/Loss", value, self.val_iteration)
