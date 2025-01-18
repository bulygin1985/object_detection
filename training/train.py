import argparse
from pathlib import Path

import pandas as pd
import torch
import torchvision
import torchvision.transforms.v2 as transforms

from data.dataset import Dataset
from models.centernet import ModelBuilder
from training.encoder import CenternetEncoder
from utils.config import IMG_HEIGHT, IMG_WIDTH, load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def criteria_builder(stop_loss, stop_epoch):
    def criteria_satisfied(current_loss, current_epoch):
        if stop_loss is not None and current_loss < stop_loss:
            return True
        if stop_epoch is not None and current_epoch >= stop_epoch:
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


def calculate_loss(model, data, batch_size=32, num_workers=0):
    batch_generator = torch.utils.data.DataLoader(
        data, num_workers=num_workers, batch_size=batch_size, shuffle=False
    )
    loss = 0.0
    count = 0
    with torch.no_grad() as ng:
        for i, data in enumerate(batch_generator):
            input_data, gt_data = data
            input_data = input_data.to(device).contiguous()

            gt_data = gt_data.to(device)
            gt_data.requires_grad = False

            loss_dict = model(input_data, gt=gt_data)
            curr_loss = loss_dict["loss"].item()
            curr_count = input_data.shape[0]
            loss += curr_loss * curr_count
            count += curr_count
    return loss / count


def train(model_conf, train_conf, data_conf):
    image_set_train = "val" if train_conf["is_overfit"] else "train"
    image_set_val = "test" if train_conf["is_overfit"] else "val"
    print(f"Selected training image_set: {image_set_train}")
    print(f"Selected validation image_set: {image_set_val}")

    dataset_val = torchvision.datasets.VOCDetection(
        root=f"../VOC",
        year="2007",
        image_set=image_set_val,
        download=data_conf["is_download"],
    )
    dataset_train = torchvision.datasets.VOCDetection(
        root=f"../VOC",
        year="2007",
        image_set=image_set_train,
        download=False,
    )

    transform = transforms.Compose(
        [
            transforms.Resize(size=(IMG_WIDTH, IMG_HEIGHT)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    encoder = CenternetEncoder(IMG_HEIGHT, IMG_WIDTH)

    dataset_val = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset_val)
    dataset_train = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset_train)
    val_data = Dataset(dataset=dataset_val, transformation=transform, encoder=encoder)
    train_data = Dataset(
        dataset=dataset_train, transformation=transform, encoder=encoder
    )
    tag = "train"
    batch_size = train_conf["batch_size"]
    train_subset_len = train_conf.get("subset_len")
    val_subset_len = train_conf.get("val_subset_len")
    num_workers = train_conf.get("num_workers", 0)

    if train_conf["is_overfit"]:
        tag = "overfit"
        assert train_subset_len is not None
        batch_size = train_subset_len
    if train_subset_len is not None:
        train_data = torch.utils.data.Subset(train_data, range(train_subset_len))
    if val_subset_len is not None:
        val_data = torch.utils.data.Subset(val_data, range(val_subset_len))

    criteria_satisfied = criteria_builder(*train_conf["stop_criteria"].values())

    model = ModelBuilder(
        filters_size=model_conf["head"]["filters_size"],
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

    batch_generator_train = torch.utils.data.DataLoader(
        train_data, num_workers=num_workers, batch_size=batch_size, shuffle=False
    )

    epoch = 1

    train_loss = []
    val_loss = []

    calculate_epoch_loss = train_conf.get("calculate_epoch_loss")

    while True:
        loss_dict = {}
        for i, data in enumerate(batch_generator_train):
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

        if calculate_epoch_loss:
            train_loss.append(
                calculate_loss(model, train_data, batch_size, num_workers)
            )
            val_loss.append(calculate_loss(model, val_data, batch_size, num_workers))
            print(f"= = = = = = = = = =")
            print(
                f"Epoch {epoch} train loss = {train_loss[-1]}, val loss = {val_loss[-1]}"
            )
            print(f"= = = = = = = = = =")

        if criteria_satisfied(loss, epoch):
            break

        check_loss_value = train_loss[-1] if calculate_epoch_loss else loss

        scheduler.step(check_loss_value)
        epoch += 1

    save_model(
        model,
        model_conf["weights_path"],
        tag=tag,
        backbone=model_conf["backbone"]["name"],
    )

    loss_df = pd.DataFrame({"train_loss": train_loss, "val_loss": val_loss})
    loss_df.to_csv("losses.csv")


if __name__ == "__main__":
    main()
