import argparse

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils import data
from callbacks.tensorboard_callback import TensorBoardCallback

from data.dataset import Dataset
from models.centernet import ModelBuilder, input_height, input_width
from training.encoder import CenternetEncoder

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overfit", action="store_true", help="overfit to 10 images")
args = parser.parse_args()

overfit = args.overfit


image_set = "val" if overfit else "train"

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

training_data = torch_dataset
lr = 0.03
batch_size = 32
patience = 7
min_lr = 1e-3


def criteria_satisfied(_, current_epoch):
    if current_epoch >= 10000:
        return True
    return False


if overfit:
    subset_len = 10
    training_data = torch.utils.data.Subset(torch_dataset, range(subset_len))
    batch_size = subset_len
    lr = 5e-2
    patience = 50
    min_lr = 1e-3

    def criteria_satisfied(current_loss, _):
        if current_loss < 1.0:
            return True
        return False


print(f"Selected image_set: {image_set}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelBuilder(alpha=0.25).to(device)

parameters = list(model.parameters())
optimizer = torch.optim.Adam(parameters, lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=patience,
    threshold=1e-4,
    threshold_mode="rel",
    cooldown=1,
    min_lr=min_lr,
)

model.train(True)

batch_generator = torch.utils.data.DataLoader(
    training_data, num_workers=0, batch_size=batch_size, shuffle=False
)

tensorboard_callback = TensorBoardCallback(
    model_name="centernet",  # або інша назва моделі
    experiment_name=None,  # автоматично створить timestamp
    log_every_n_steps=10,
    log_images_every_n_epochs=1
)

hparams = {
    "learning_rate": lr,
    "batch_size": batch_size,
    "patience": patience,
    "min_lr": min_lr,
    "alpha": model.alpha if hasattr(model, 'alpha') else None,
}
tensorboard_callback.log_hyperparameters(hparams)

tensorboard_callback.log_model_graph(
    model,
    input_size=(batch_size, 3, input_height, input_width)
)

epoch = 1
get_desired_loss = False

while True:
    print("EPOCH {}:".format(epoch))
    epoch_losses = []

    for batch_idx, data in enumerate(batch_generator):
        input_data, gt_data = data
        input_data = input_data.to(device).contiguous()
        gt_data = gt_data.to(device)
        gt_data.requires_grad = False

        loss_dict = model(input_data, gt=gt_data)
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        # Логуємо метрики батчу
        tensorboard_callback.on_train_batch_end(
            batch_idx,
            loss_dict,
            model,
            optimizer
        )

        epoch_losses.append(loss_dict["loss"].item())
        print(f" loss={loss_dict['loss']}, lr={scheduler.get_last_lr()}")

    # Логуємо метрики епохи
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    tensorboard_callback.on_epoch_end(
        epoch,
        model,
        avg_epoch_loss,
        input_data  # Передаємо останній батч для візуалізації
    )

    if criteria_satisfied(loss_dict["loss"], epoch):
        break

    scheduler.step(loss_dict["loss"])
    epoch += 1

torch.save(model.state_dict(), "../models/checkpoints/pretrained_weights.pt")
