import argparse
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.tensorboard import SummaryWriter

from callbacks.save_checkpoint import SaveCheckPoint
from data.dataset import Dataset
from data.dataset_loaders import MSCOCODatasetLoader
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


def log_stats(tensorboard_writer, epoch, lr, losses: dict):
    train_validation_loss = losses["validation"]["train"]
    val_validation_loss = losses["validation"]["val"]
    # Write to Tensorboard
    tensorboard_writer.add_scalar("Train/loss", train_validation_loss, epoch)
    tensorboard_writer.add_scalar("Val/loss", val_validation_loss, epoch)
    tensorboard_writer.add_scalar("Train/lr", lr, epoch)


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


def compose_transforms(data_augmentation_params=None):
    transforms_list = [transforms.Resize(size=(IMG_WIDTH, IMG_HEIGHT))]
    if data_augmentation_params:
        resize_crop_scale = data_augmentation_params.get(
            "random_resize_crop_scale", None
        )
        resize_crop_ratio = data_augmentation_params.get(
            "random_resize_crop_ratio", None
        )
        if resize_crop_scale is not None or resize_crop_ratio is not None:
            print(
                f"Applying RandomResizedCrop scale={resize_crop_scale} ratio={resize_crop_ratio}"
            )
            transforms_list = [
                transforms.RandomResizedCrop(
                    size=(IMG_WIDTH, IMG_HEIGHT),
                    scale=resize_crop_scale,
                    ratio=resize_crop_ratio,
                )
            ]
        if data_augmentation_params.get("random_flip_horizontal"):
            print("Applying random_flip_horizontal")
            transforms_list.append(transforms.RandomHorizontalFlip())
        brightness_jitter = data_augmentation_params.get("color_jitter_brightness")
        if brightness_jitter:
            print(f"Applying color_jitter_brightness={brightness_jitter}")
            transforms_list.append(transforms.ColorJitter(brightness=brightness_jitter))
    return transforms.Compose(
        transforms_list
        + [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )


def calculate_validation_loss(
    model, data, batch_size=32, num_workers=0, pin_memory=False
):
    batch_generator = torch.utils.data.DataLoader(
        data,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    loss = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
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


def train(config_filepath):
    model_conf, train_conf, data_conf = load_config(config_filepath)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"runs/training_{timestamp}"
    os.makedirs(run_folder)
    shutil.copy(config_filepath, run_folder)
    writer = SummaryWriter(run_folder)

    image_set_train = "val" if train_conf["is_overfit"] else "train"
    image_set_val = "test" if train_conf["is_overfit"] else "val"
    print(f"Selected training image_set: {image_set_train}")
    print(f"Selected validation image_set: {image_set_val}")

    train_ds_loader = MSCOCODatasetLoader(
        data_conf[image_set_train]["images_folder"],
        data_conf[image_set_train]["ann_file"],
    )
    val_ds_loader = MSCOCODatasetLoader(
        data_conf[image_set_val]["images_folder"],
        data_conf[image_set_val]["ann_file"],
    )
    train_transform = compose_transforms(train_conf.get("data_augmentation"))
    val_transform = compose_transforms()
    encoder = CenternetEncoder(
        IMG_HEIGHT, IMG_WIDTH, n_classes=data_conf.get("class_amount", 20)
    )

    train_data = Dataset(
        dataset=train_ds_loader.get_dataset(),
        transformation=train_transform,
        encoder=encoder,
    )
    val_data = Dataset(
        dataset=val_ds_loader.get_dataset(),
        transformation=val_transform,
        encoder=encoder,
    )

    tag = "train"
    batch_size = train_conf["batch_size"]
    train_subset_len = train_conf.get("subset_len")
    val_subset_len = train_conf.get("val_subset_len")
    num_workers = train_conf.get("num_workers", 0)

    if train_conf["is_overfit"]:
        tag = "overfit"
        batch_size = train_subset_len
    if train_subset_len is not None:
        train_data = torch.utils.data.Subset(train_data, range(train_subset_len))
    if val_subset_len is not None:
        val_data = torch.utils.data.Subset(val_data, range(val_subset_len))

    criteria_satisfied = criteria_builder(*train_conf["stop_criteria"].values())

    model = ModelBuilder(
        filters_size=model_conf["head"]["filters_size"],
        alpha=model_conf["alpha"],
        class_number=data_conf.get("class_amount", 20),
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

    pin_memory = train_conf.get("pin_memory", False)
    batch_generator_train = torch.utils.data.DataLoader(
        train_data,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=train_conf.get("drop_last", False),
        pin_memory=pin_memory,
        shuffle=train_conf.get("shuffle", False),
    )

    epoch = 1

    train_loss_history = []
    val_loss_history = []
    best_val_loss_history = []
    best_val_loss = float("inf")

    calculate_epoch_loss = train_conf.get("calculate_epoch_loss")
    save_best_model = train_conf.get("save_best_model", False)
    save_best_model_skip_epochs = train_conf.get("save_best_model_skip_epochs", 0)
    checkpoint_callback = None
    if save_best_model:
        checkpoint_callback = SaveCheckPoint(
            model,
            run_folder,
            monitor="val_loss",
            best_mode="min",
            skip_epochs=save_best_model_skip_epochs,
        )

    while True:
        epoch_start = time.perf_counter()
        model.train()
        for i, data in enumerate(batch_generator_train):
            input_data, gt_data = data
            input_data = input_data.to(device).contiguous()

            gt_data = gt_data.to(device)
            gt_data.requires_grad = False

            loss_dict = model(input_data, gt=gt_data)
            optimizer.zero_grad()
            loss_dict["loss"].backward()

            optimizer.step()
            loss = loss_dict["loss"].item()
            curr_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}, batch {i}, loss={loss:.3f}, lr={curr_lr}")

        print("= = = = = = = = = =")
        if calculate_epoch_loss or save_best_model:
            last_lr = scheduler.get_last_lr()[0]
            train_validation_loss = calculate_validation_loss(
                model, train_data, batch_size, num_workers, pin_memory
            )
            val_validation_loss = calculate_validation_loss(
                model, val_data, batch_size, num_workers, pin_memory
            )
            train_loss_history.append(train_validation_loss)
            val_loss_history.append(val_validation_loss)
            if val_validation_loss < best_val_loss:
                best_val_loss = val_validation_loss
            best_val_loss_history.append(best_val_loss)

            loss_stats = {
                "validation": {
                    "train": train_validation_loss,
                    "val": val_validation_loss,
                }
            }
            log_stats(writer, epoch, last_lr, loss_stats)
            print(
                (
                    f"Epoch {epoch} train loss = {train_validation_loss:.4f}, "
                    f"val loss = {val_validation_loss:.4f}, "
                    f"best val loss = {best_val_loss:.4f}"
                )
            )
            if checkpoint_callback is not None:
                checkpoint_callback.on_epoch_end(
                    epoch, {"val_loss": val_validation_loss}
                )
        print(
            f"Epoch calculation time is {time.perf_counter()-epoch_start:.2f} seconds"
        )
        print("= = = = = = = = = =")
        if criteria_satisfied(loss, epoch):
            break

        check_loss_value = train_validation_loss if calculate_epoch_loss else loss

        scheduler.step(check_loss_value)
        epoch += 1

    writer.close()

    save_model(
        model,
        run_folder,
        tag=tag,
        backbone=model_conf["backbone"]["name"],
    )

    if model_conf["weights_path"]:
        save_model(
            model,
            model_conf["weights_path"],
            tag=tag,
            backbone=model_conf["backbone"]["name"],
        )

    loss_df = pd.DataFrame(
        {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "best_val_loss": best_val_loss_history,
        }
    )
    loss_df.to_csv(os.path.join(run_folder, "losses.csv"))


def main(config_path: str = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to config file")
    args = parser.parse_args()

    filepath = args.config or config_path
    train(filepath)


if __name__ == "__main__":
    main("config_example_quick_train_with_epoch_loss.json")
