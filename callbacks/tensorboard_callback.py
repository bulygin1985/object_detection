import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class AdvancedTensorBoardCallback:
    def __init__(
        self,
        log_dir: str = "training/runs",
        experiment_name: Optional[str] = None,
        enabled_features: Optional[List[str]] = None,
    ):
        """
        Розширений TensorBoard callback з можливістю вибору метрик для відстеження.

        Args:
            log_dir: Базова директорія для логів
            experiment_name: Назва експерименту (якщо None, використовується timestamp)
            enabled_features: Список активних фіч ['metrics', 'histograms', 'images',
                                                'gradients', 'weights', 'embeddings']
        """
        # Налаштування директорії логів
        self.base_log_dir = Path(log_dir)
        self.experiment_name = experiment_name or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.log_dir = self.base_log_dir / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Ініціалізація writer'а
        self.writer = SummaryWriter(str(self.log_dir))

        # Налаштування активних фіч
        all_features = {
            "metrics",
            "histograms",
            "images",
            "gradients",
            "weights",
            "embeddings",
        }
        self.enabled_features = (
            set(enabled_features) if enabled_features else all_features
        )

        # Лічильники та акумулятори
        self.epoch_start_time = None
        self.global_step = 0
        self.best_metrics = {}
        self.metrics_history = {}

    def on_training_start(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ):
        """Логування початкової конфігурації тренування"""
        # Зберігаємо гіперпараметри
        hparams = {
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "model_name": model.__class__.__name__,
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
        self.writer.add_hparams(hparams, {"status": 0})

    def on_epoch_start(self):
        """Відмічаємо початок епохи"""
        self.epoch_start_time = time.time()

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        input_example: Optional[torch.Tensor] = None,
    ):
        """
        Логування метрик в кінці епохи

        Args:
            epoch: Номер поточної епохи
            model: Модель PyTorch
            optimizer: Оптимізатор
            metrics: Словник метрик
            input_example: Приклад вхідних даних для логування графу моделі
        """
        # Базові метрики
        if "metrics" in self.enabled_features:
            self._log_metrics(metrics, epoch)
            self._log_learning_rate(optimizer, epoch)
            if self.epoch_start_time:
                epoch_time = time.time() - self.epoch_start_time
                self.writer.add_scalar("Time/epoch_duration", epoch_time, epoch)

        # Гістограми та розподіли
        if "histograms" in self.enabled_features:
            self._log_histograms(model, epoch)

        # Градієнти
        if "gradients" in self.enabled_features:
            self._log_gradients(model, epoch)

        # Ваги моделі
        if "weights" in self.enabled_features:
            self._log_model_weights(model, epoch)

        # Граф моделі (тільки раз на тренування)
        if input_example is not None and epoch == 0:
            self.writer.add_graph(model, input_example)

        # Оновлюємо найкращі метрики
        self._update_best_metrics(metrics)

        # Флашимо дані
        self.writer.flush()

    def _log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Логування метрик з розбивкою по категоріях"""
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            # Зберігаємо історію
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)

            # Логуємо поточне значення
            self.writer.add_scalar(f"Metrics/{name}", value, epoch)

            # Логуємо running average
            if len(self.metrics_history[name]) > 1:
                avg = np.mean(
                    self.metrics_history[name][-5:]
                )  # середнє за останні 5 епох
                self.writer.add_scalar(f"Metrics/{name}_avg", avg, epoch)

    def _log_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        """Логування learning rate для кожної групи параметрів"""
        for idx, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(
                f"Learning_rate/group_{idx}", param_group["lr"], epoch
            )

    def _log_histograms(self, model: torch.nn.Module, epoch: int):
        """Логування гістограм активацій та ваг"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"Weights/{name}", param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    def _log_gradients(self, model: torch.nn.Module, epoch: int):
        """Логування статистик градієнтів"""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                self.writer.add_scalar(f"Gradients/mean/{name}", grad.mean(), epoch)
                self.writer.add_scalar(f"Gradients/std/{name}", grad.std(), epoch)
                self.writer.add_scalar(f"Gradients/norm/{name}", grad.norm(), epoch)

    def _log_model_weights(self, model: torch.nn.Module, epoch: int):
        """Логування статистик ваг моделі"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_scalar(f"Weights/mean/{name}", param.data.mean(), epoch)
                self.writer.add_scalar(f"Weights/std/{name}", param.data.std(), epoch)
                self.writer.add_scalar(f"Weights/norm/{name}", param.data.norm(), epoch)

    def _update_best_metrics(self, metrics: Dict[str, float]):
        """Оновлення найкращих значень метрик"""
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            if name not in self.best_metrics:
                self.best_metrics[name] = value
            else:
                # Припускаємо, що більше значення краще
                self.best_metrics[name] = max(self.best_metrics[name], value)

            self.writer.add_scalar(
                f"Best/{name}", self.best_metrics[name], self.global_step
            )

    def log_custom_metric(
        self, name: str, value: Union[float, torch.Tensor], step: Optional[int] = None
    ):
        """Логування довільної метрики"""
        if isinstance(value, torch.Tensor):
            value = value.item()

        step = step if step is not None else self.global_step
        self.writer.add_scalar(f"Custom/{name}", value, step)

    def log_batch_metrics(self, metrics: Dict[str, float], batch_idx: int, epoch: int):
        """Логування метрик на рівні батчів"""
        step = epoch * 1000 + batch_idx  # припускаємо максимум 1000 батчів на епоху
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f"Batch/{name}", value, step)

    def close(self):
        """Закриття writer'а та збереження підсумкових метрик"""
        # Зберігаємо фінальні метрики
        for name, values in self.metrics_history.items():
            self.writer.add_scalar(f"Final/{name}_mean", np.mean(values), 0)
            self.writer.add_scalar(f"Final/{name}_std", np.std(values), 0)

        self.writer.close()
