import argparse
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.v2 as transforms

from data.dataset import Dataset
from models.centernet import ModelBuilder, input_height, input_width
from training.encoder import CenternetEncoder
from utils.config import load_config


def criteria_builder(stop_loss, stop_epoch):
    def criteria_satisfied(current_loss, current_epoch):
        if stop_loss is not None and current_loss < 1.0:
            return True
        if stop_epoch is not None and current_epoch > stop_epoch:
            return True
        return False

    return criteria_satisfied


def save_model(model, weights_path: str = None, **kwargs):
    checkpoints_dir = weights_path or "models/checkpoints"
    tag = kwargs.get("tag", "train")
    backbone = kwargs.get("backbone", "default")
    cur_dir = Path(__file__).resolve().parent

    checkpoint_filename = (
        cur_dir.parent / checkpoints_dir / f"pretrained_weights_{tag}_{backbone}.pt"
    )

    torch.save(model.state_dict(), checkpoint_filename)
    print(f"Saved model checkpoint to {checkpoint_filename}")


def main(config_path: str = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to config file")
    args = parser.parse_args()

    filepath = args.config or config_path

    model_conf, train_conf, data_conf = load_config(filepath)

    train(model_conf, train_conf, data_conf)


def train(model_conf, train_conf, data_conf):
    image_set = "val" if train_conf["is_overfit"] else "train"
    print(f"Selected image_set: {image_set}")

    dataset_val = torchvision.datasets.VOCDetection(
        root="../VOC",
        year="2007",
        image_set=image_set,
        download=data_conf["is_download"],
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
    torch_dataset = Dataset(
        dataset=dataset_val, transformation=transform, encoder=encoder
    )
    tag = "train"
    training_data = torch_dataset
    batch_size = train_conf["batch_size"]

    if train_conf["is_overfit"]:
        tag = "overfit"
        training_data = torch.utils.data.Subset(
            torch_dataset, range(train_conf["subset_len"])
        )
        batch_size = train_conf["subset_len"]

    criteria_satisfied = criteria_builder(*train_conf["stop_criteria"].values())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBuilder(
        alpha=model_conf["alpha"],
        backbone=model_conf["backbone"]["name"],
        backbone_weights=model_conf["backbone"]["pretrained_weights"],
    ).to(device)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=train_conf["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_conf["lr_schedule"]["factor"],
        patience=train_conf["lr_schedule"]["patience"],
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=1,
        min_lr=train_conf["lr_schedule"]["min_lr"],
    )

    model.train(True)

    batch_generator = torch.utils.data.DataLoader(
        training_data, num_workers=0, batch_size=batch_size, shuffle=False
    )

    epoch = 1
    get_desired_loss = False

    while True:
        loss_dict = {}
        for i, data in enumerate(batch_generator):
            input_data, gt_data = data
            input_data = input_data.to(device).contiguous()

            gt_data = gt_data.to(device)
            gt_data.requires_grad = False

            loss_dict = model(input_data, gt=gt_data)
            optimizer.zero_grad()  # compute gradient and do optimize step
            loss_dict["loss"].backward()

            optimizer.step()
            loss = loss_dict["loss"].item()
            curr_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}, batch {i}, loss={loss:.3f}, lr={curr_lr}")

        if criteria_satisfied(loss, epoch):
            break

        scheduler.step(loss_dict["loss"])
        epoch += 1

    save_model(
        model,
        model_conf["weights_path"],
        tag=tag,
        backbone=model_conf["backbone"]["name"],
    )


if __name__ == "__main__":
    main()
