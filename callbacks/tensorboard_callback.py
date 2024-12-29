import os
from datetime import datetime
from typing import Dict, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardCallback:
    """
    Callback для логування метрик тренування CenterNet та візуалізацій у TensorBoard.
    Підтримує логування для кількох моделей одночасно.
    """

    def __init__(
            self,
            model_name: str,
            experiment_name: Optional[str] = None,
            log_every_n_steps: int = 10,
            log_images_every_n_epochs: int = 1,
    ):
        """
        Ініціалізація TensorBoard callback.

        Args:
            model_name: Назва моделі (наприклад, 'centernet', 'centernet_large')
            experiment_name: Назва експерименту. Якщо не вказано, використовується timestamp
            log_every_n_steps: Частота логування метрик під час тренування
            log_images_every_n_epochs: Частота логування прикладів детекцій
        """
        # Створюємо структуру директорій для логів
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join("logs", model_name, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        self.model_name = model_name
        self.log_every_n_steps = log_every_n_steps
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.global_step = 0
        self.current_epoch = 0

    def on_train_batch_end(
            self,
            batch_idx: int,
            loss_dict: Dict[str, torch.Tensor],
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Логування метрик після кожного батчу.

        Args:
            batch_idx: Індекс поточного батчу
            loss_dict: Словник з компонентами функції втрат
            model: Модель CenterNet
            optimizer: Оптимізатор
        """
        if batch_idx % self.log_every_n_steps == 0:
            # Логуємо всі компоненти loss
            for loss_name, loss_value in loss_dict.items():
                self.writer.add_scalar(
                    f"{self.model_name}/train/batch/{loss_name}",
                    loss_value.item(),
                    self.global_step
                )

            # Логуємо learning rate
            for i, param_group in enumerate(optimizer.param_groups):
                self.writer.add_scalar(
                    f"{self.model_name}/train/batch/learning_rate/group_{i}",
                    param_group["lr"],
                    self.global_step
                )

        self.global_step += 1

    def on_epoch_end(
            self,
            epoch: int,
            model: torch.nn.Module,
            epoch_loss: float,
            input_batch: Optional[torch.Tensor] = None,
            validation_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Логування метрик та візуалізацій в кінці кожної епохи.

        Args:
            epoch: Номер поточної епохи
            model: Модель CenterNet
            epoch_loss: Середній loss за епоху
            input_batch: Опціональний батч зображень для візуалізації
            validation_metrics: Опціональний словник з метриками валідації
        """
        self.current_epoch = epoch

        # Логуємо метрики рівня епохи
        self.writer.add_scalar(f"{self.model_name}/train/epoch/loss", epoch_loss, epoch)

        # Логуємо валідаційні метрики
        if validation_metrics:
            for metric_name, metric_value in validation_metrics.items():
                self.writer.add_scalar(
                    f"{self.model_name}/validation/epoch/{metric_name}",
                    metric_value,
                    epoch
                )

        # Логуємо параметри моделі як гістограми
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(
                    f"{self.model_name}/parameters/{name}",
                    param,
                    epoch
                )
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"{self.model_name}/gradients/{name}",
                        param.grad,
                        epoch
                    )

        # Логуємо приклади детекцій
        if input_batch is not None and epoch % self.log_images_every_n_epochs == 0:
            with torch.no_grad():
                model.eval()
                predictions = model(input_batch)
                model.train()

                # Логуємо перші кілька зображень з їх передбаченнями
                for idx in range(min(3, input_batch.size(0))):
                    self.writer.add_image(
                        f"{self.model_name}/detections/image_{idx}",
                        input_batch[idx] * 0.5 + 0.5,  # Денормалізація
                        epoch
                    )
                    # Додаємо візуалізацію heatmap
                    if predictions.size(1) > 4:  # Якщо передбачення включають heatmaps
                        heatmap = predictions[idx, :predictions.size(1) - 4].max(dim=0)[0]
                        self.writer.add_image(
                            f"{self.model_name}/detections/heatmap_{idx}",
                            heatmap.unsqueeze(0),
                            epoch
                        )

    def on_train_end(self) -> None:
        """Очищення при завершенні тренування."""
        self.writer.close()

    def log_model_graph(self, model: torch.nn.Module, input_size: tuple) -> None:
        """
        Логування графу архітектури моделі в TensorBoard.

        Args:
            model: Модель CenterNet
            input_size: Розмір вхідного тензору (batch_size, channels, height, width)
        """
        dummy_input = torch.zeros(input_size, device=next(model.parameters()).device)
        self.writer.add_graph(model, dummy_input)

    def log_hyperparameters(self, hparams: Dict) -> None:
        """
        Логування гіперпараметрів моделі.

        Args:
            hparams: Словник з гіперпараметрами
        """
        self.writer.add_hparams(hparams, {})