import argparse
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils import data

from callbacks.model_save import SaveBestModelCallback
from callbacks.tensorboard_callback import AdvancedTensorBoardCallback
from data.dataset import Dataset
from models.centernet import ModelBuilder, input_height, input_width
from training.encoder import CenternetEncoder

# Налаштування аргументів
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overfit", action="store_true", help="overfit to 10 images")
args = parser.parse_args()

overfit = args.overfit
image_set = "val" if overfit else "train"

# Налаштування пристрою
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Налаштування датасету
dataset_val = torchvision.datasets.VOCDetection(
    root="../VOC", year="2007", image_set=image_set, download=False
)

transform = transforms.Compose(
    [
        transforms.Resize(size=(input_width, input_height)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

encoder = CenternetEncoder(input_height, input_width)
dataset_val = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset_val)
torch_dataset = Dataset(dataset=dataset_val, transformation=transform, encoder=encoder)

# Налаштування параметрів тренування
training_data = torch_dataset
lr = 1e-3
batch_size = 32
patience = 7
min_lr = 1e-3
max_epochs = 10000
loss_threshold = 1.0

# Модифікація параметрів для overfit режиму
if overfit:
    subset_len = 10
    training_data = torch.utils.data.Subset(torch_dataset, range(subset_len))
    batch_size = subset_len
    lr = 5e-2
    patience = 50
    min_lr = 1e-5

print(f"Selected image_set: {image_set}")
print(f"Dataset size: {len(training_data)}")

# Ініціалізація моделі та оптимізатора
model = ModelBuilder(alpha=0.25).to(device)
parameters = list(model.parameters())
optimizer = torch.optim.Adam(parameters, lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,
    patience=patience,
    threshold=1e-4,
    threshold_mode="rel",
    cooldown=1,
    min_lr=min_lr,
)

# Ініціалізація callbacks
save_callback = SaveBestModelCallback(
    save_dir="../callbacks/checkpoints",
    metric_name="loss",
    greater_is_better=False,
    start_saving_threshold=5.0,
    min_improvement=0.5,
)

tensorboard_callback = AdvancedTensorBoardCallback(
    log_dir="runs",
    experiment_name=f"centernet_{'overfit_' if overfit else ''}{time.strftime('%Y%m%d_%H%M%S')}",
    enabled_features=["metrics", "histograms", "gradients", "weights"],
)

# Налаштування DataLoader
batch_generator = torch.utils.data.DataLoader(
    training_data, num_workers=0, batch_size=batch_size, shuffle=False
)

# Початок тренування
model.train(True)
tensorboard_callback.on_training_start(model, optimizer)

epoch = 1
total_batches = len(batch_generator)

while epoch <= max_epochs:
    print(f"\nEPOCH {epoch}:")
    epoch_start_time = time.time()
    tensorboard_callback.on_epoch_start()

    total_loss = 0
    total_heatmap_loss = 0
    total_size_loss = 0
    total_offset_loss = 0

    for batch_idx, data in enumerate(batch_generator):
        # Підготовка даних
        input_data, gt_data = data
        input_data = input_data.to(device).contiguous()
        gt_data = gt_data.to(device)
        gt_data.requires_grad = False

        # Forward pass
        loss_dict = model(input_data, gt=gt_data)

        # Backward pass
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        # Акумуляція втрат
        total_loss += loss_dict["loss"].item()
        total_heatmap_loss += loss_dict.get("heatmap_loss", 0)
        total_size_loss += loss_dict.get("size_loss", 0)
        total_offset_loss += loss_dict.get("offset_loss", 0)

        # Логування метрик батча
        batch_metrics = {
            "batch_loss": loss_dict["loss"].item(),
            "batch_heatmap_loss": loss_dict.get("heatmap_loss", 0),
            "batch_size_loss": loss_dict.get("size_loss", 0),
            "batch_offset_loss": loss_dict.get("offset_loss", 0),
        }
        tensorboard_callback.log_batch_metrics(batch_metrics, batch_idx, epoch)

        # Прогрес
        if batch_idx % 100 == 0:
            print(
                f"Batch [{batch_idx}/{total_batches}], "
                f"Loss: {loss_dict['loss']:.4f}, "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    # Обчислення середніх метрик за епоху
    avg_loss = total_loss / total_batches
    avg_heatmap_loss = total_heatmap_loss / total_batches
    avg_size_loss = total_size_loss / total_batches
    avg_offset_loss = total_offset_loss / total_batches

    # Час виконання епохи
    epoch_time = time.time() - epoch_start_time

    # Формування метрик епохи
    epoch_metrics = {
        "loss": avg_loss,
        "heatmap_loss": avg_heatmap_loss,
        "size_loss": avg_size_loss,
        "offset_loss": avg_offset_loss,
        "learning_rate": scheduler.get_last_lr()[0],
        "epoch_time": epoch_time,
    }

    # Логування метрик епохи
    tensorboard_callback.on_epoch_end(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        metrics=epoch_metrics,
        input_example=input_data[:1],
    )

    # Збереження кращої моделі
    save_callback.on_eval_epoch_end(
        model=model, optimizer=optimizer, epoch=epoch, current_metric=avg_loss
    )

    # Оновлення scheduler
    scheduler.step(avg_loss)

    # Перевірка умов зупинки
    if avg_loss < loss_threshold:
        print(f"\nDosягнуто цільове значення loss ({loss_threshold})")
        break

    # Інкремент епохи
    epoch += 1

# Збереження фінальної моделі
save_path = Path("../models/checkpoints/pretrained_weights.pt")
save_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), save_path)

# Закриття TensorBoard
tensorboard_callback.close()

print("\nТренування завершено!")
print(f"Фінальний loss: {avg_loss:.4f}")
print(f"Модель збережено в: {save_path}")
