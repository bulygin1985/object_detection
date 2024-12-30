import os

import torch
from pytorch_accelerated.callbacks import TrainerCallback


class SaveBestModelCallback(TrainerCallback):
    def __init__(
        self,
        save_dir: str,
        metric_name: str = "loss",
        greater_is_better: bool = False,
        start_saving_threshold: float = None,
        min_improvement: float = 0.0,
    ):
        """
        Args:
        save_dir: Directory for saving the model weights
        metric_name: Name of the metric to track
        greater_is_better: True if a higher metric value is better
        start_saving_threshold: Metric value at which model saving begins.
                                For loss (greater_is_better=False) - save when the metric is less than the threshold
                                For accuracy (greater_is_better=True) - save when the metric is greater than the threshold
        min_improvement: The minimum difference between the best saved metric and the current metric
                         required to save a new model
        """
        super().__init__()
        self.save_dir = save_dir
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.start_saving_threshold = start_saving_threshold
        self.min_improvement = min_improvement

        self.best_metric = float("inf")
        if greater_is_better:
            self.best_metric = float("-inf")

        os.makedirs(save_dir, exist_ok=True)

    def _is_better(self, current_metric: float) -> bool:
        if self.greater_is_better:
            return (current_metric - self.best_metric) >= self.min_improvement
        return (self.best_metric - current_metric) >= self.min_improvement

    def _should_start_saving(self, current_metric: float) -> bool:
        if self.start_saving_threshold is None:
            return True

        if self.greater_is_better:
            return current_metric >= self.start_saving_threshold
        return current_metric <= self.start_saving_threshold

    def _clean_old_checkpoints(self):
        for file in os.listdir(self.save_dir):
            if file.startswith("best_model_"):
                os.remove(os.path.join(self.save_dir, file))

    def on_eval_epoch_end(self, model, optimizer, epoch, current_metric):
        if not self._should_start_saving(current_metric):
            return

        if self._is_better(current_metric):
            self.best_metric = current_metric

            self._clean_old_checkpoints()

            save_path = os.path.join(
                self.save_dir, f"best_model_{self.metric_name}_{current_metric:.4f}.pt"
            )

            torch.save(model.state_dict(), save_path)
            print(f"\nsaved {self.metric_name}={current_metric:.4f}")