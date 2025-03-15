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
from training.train_utils import *
from utils.config import IMG_HEIGHT, IMG_WIDTH, load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def criteria_builder(loss, epoch):
    def criteria_satisfied(current_loss, current_epoch):
        if loss is not None and current_loss < loss:
            return True
        if epoch is not None and current_epoch >= epoch:
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


def extend_relative_path_to_parent(path):
    if os.path.isabs(path):
        return path
    cur_dir = Path(__file__).resolve().parent
    return os.path.join(cur_dir.parent, path)


def save_model(model, weights_path: str, tag: str = "train", backbone: str = "default"):
    checkpoint_filename = os.path.join(
        weights_path, f"pretrained_weights_{tag}_{backbone}.pt"
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


def calculate_loss_on_batch_generator(model, batch_generator):
    loss = 0.0
    count = 0
    model.eval()
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
    return calculate_loss_on_batch_generator(model, batch_generator)


def train(config_filepath):
    model_conf, train_conf, data_conf = load_config(config_filepath)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = extend_relative_path_to_parent(f"runs/training_{timestamp}")
    print("run folder: ", run_folder)
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
        assert train_subset_len is not None
        batch_size = train_subset_len
    if train_subset_len is not None:
        train_data = torch.utils.data.Subset(train_data, range(train_subset_len))
    if val_subset_len is not None:
        val_data = torch.utils.data.Subset(val_data, range(val_subset_len))

    criteria_satisfied = criteria_builder(**train_conf["stop_criteria"])
    backbone_name = model_conf["backbone"]["name"]
    model = ModelBuilder(
        filters_size=model_conf["head"]["filters_size"],
        alpha=model_conf["alpha"],
        class_number=data_conf.get("class_amount", 20),
        backbone=backbone_name,
        backbone_weights=model_conf["backbone"]["pretrained_weights"],
    ).to(device)

    lr = train_conf["lr"]
    lr_backbone = train_conf.get("lr_backbone", lr)
    lr_head = train_conf.get("lr_head", lr)

    head_pretrain_epochs = train_conf.get("head_pretrain_epochs")

    bb_train_params_patterns_include = train_conf.get(
        "backbone_trainable_params_patterns_include"
    )
    bb_train_params_patterns_exclude = train_conf.get(
        "backbone_trainable_params_patterns_exclude"
    )
    if bb_train_params_patterns_exclude or bb_train_params_patterns_include:
        trainable_backbone_params = filter_named_values_by_pattern(
            model.backbone.named_parameters(),
            bb_train_params_patterns_include,
            bb_train_params_patterns_exclude,
        )
        print("Filter backbone trainable parameters:")
        print(f"   include: {bb_train_params_patterns_include}")
        print(f"   exclude: {bb_train_params_patterns_exclude}")
        print(
            f"   trainable {len(trainable_backbone_params)} of {len(list(model.backbone.parameters()))}"
        )
    else:
        trainable_backbone_params = model.backbone.parameters()

    lr_schedule_conf = train_conf["lr_schedule"]
    scheduler_type = lr_schedule_conf["type"]

    if head_pretrain_epochs:
        lr_head_start = train_conf.get("lr_head_pretrain", lr_head)
        lr_backbone_start = 0.0
    else:
        lr_head_start, lr_backbone_start = lr_head, lr_backbone

    weight_decay = train_conf.get("weight_decay")
    weight_decay_bias = train_conf.get("weight_decay_bias", True)
    if weight_decay > 0:
        decay_params, nodecay_params = split_params_for_weight_decay(
            model, weight_decay_bias
        )
        head_decay_params = [p for n, p in decay_params if n.startswith("head.")]
        head_nodecay_params = [p for n, p in nodecay_params if n.startswith("head.")]
        backbone_decay_params = [
            (n, p) for n, p in decay_params if n.startswith("backbone.")
        ]
        backbone_nodecay_params = [
            (n, p) for n, p in nodecay_params if n.startswith("backbone.")
        ]
        if bb_train_params_patterns_exclude or bb_train_params_patterns_include:
            backbone_decay_params = filter_named_values_by_pattern(
                backbone_decay_params,
                bb_train_params_patterns_include,
                bb_train_params_patterns_exclude,
            )
            backbone_nodecay_params = filter_named_values_by_pattern(
                backbone_nodecay_params,
                bb_train_params_patterns_include,
                bb_train_params_patterns_exclude,
            )
        else:
            backbone_nodecay_params = [p for n, p in backbone_nodecay_params]
            backbone_decay_params = [p for n, p in backbone_decay_params]
        opt_params = [
            {
                "params": backbone_decay_params,
                "lr": lr_backbone_start,
                "weight_decay": weight_decay,
            },
            {
                "params": backbone_nodecay_params,
                "lr": lr_backbone_start,
                "weight_decay": 0.0,
            },
            {
                "params": head_decay_params,
                "lr": lr_head_start,
                "weight_decay": weight_decay,
            },
            {"params": head_nodecay_params, "lr": lr_head_start, "weight_decay": 0.0},
        ]
        print(f"applying weight decay = {weight_decay}")
    else:
        opt_params = [
            {"params": trainable_backbone_params, "lr": lr_backbone_start},
            {"params": model.head.parameters(), "lr": lr_head_start},
        ]
    optimizer_type = optimizer_type_by_str(train_conf.get("optimizer", "Adam"))
    print(f"using {optimizer_type} optimizer")
    optimizer = optimizer_type(opt_params, lr=0.0)

    model.train(True)
    persistent_workers = train_conf.get("persistent_workers", False)
    pin_memory = train_conf.get("pin_memory", False)
    batch_generator_train = torch.utils.data.DataLoader(
        train_data,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=persistent_workers,
        drop_last=train_conf.get("drop_last"),
    )

    epoch = 1

    train_loss_history = []
    val_loss_history = []
    best_val_loss_history = []
    lr_head_history = []
    lr_backbone_history = []
    best_val_loss = float("inf")

    calculate_epoch_loss = train_conf.get("calculate_epoch_loss")
    save_best_model = train_conf.get("save_best_model", True)
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

    num_workers_validation = train_conf.get("num_workers_validation", num_workers)
    batch_size_val = train_conf.get("batch_size_val", batch_size)
    warmup_rate_scale = train_conf.get("warmup_rate_scale")
    warmup_epochs = train_conf.get("warmup_epochs", 0)
    batch_size_val = train_conf.get("batch_size_val", batch_size)

    if calculate_epoch_loss:
        batch_generator_val = torch.utils.data.DataLoader(
            val_data,
            num_workers=num_workers_validation,
            batch_size=batch_size_val,
            shuffle=False,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )
        batch_generator_train_val = torch.utils.data.DataLoader(
            train_data,
            num_workers=num_workers_validation,
            batch_size=batch_size_val,
            shuffle=False,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

    while True:
        epoch_start = time.perf_counter()
        pretrain = head_pretrain_epochs and epoch <= head_pretrain_epochs
        if warmup_epochs and epoch <= warmup_epochs + 1:
            # switch optimizer LRs
            warmup_ratio = 1 - (1 - warmup_rate_scale) / warmup_epochs * (
                warmup_epochs + 1 - epoch
            )
            optimizer.param_groups[0]["lr"] = lr_backbone_start * warmup_ratio
            optimizer.param_groups[-1]["lr"] = lr_head_start * warmup_ratio
            if weight_decay > 0:
                optimizer.param_groups[1]["lr"] = optimizer.param_groups[0]["lr"]
                optimizer.param_groups[2]["lr"] = optimizer.param_groups[-1]["lr"]
        if not pretrain and epoch == (head_pretrain_epochs + 1):
            if head_pretrain_epochs:
                # switch optimizer LRs
                if weight_decay > 0:
                    optimizer.param_groups[0]["lr"] = lr_backbone
                    optimizer.param_groups[1]["lr"] = lr_backbone
                    optimizer.param_groups[2]["lr"] = lr_head
                    optimizer.param_groups[3]["lr"] = lr_head
                else:
                    optimizer.param_groups[0]["lr"] = lr_backbone
                    optimizer.param_groups[1]["lr"] = lr_head
            scheduler = create_scheduler(optimizer, lr_schedule_conf)
        model.train()
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
            curr_lr = [optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]]
            if weight_decay > 0:
                curr_lr += [
                    optimizer.param_groups[2]["lr"],
                    optimizer.param_groups[3]["lr"],
                ]
            lr_to_show = curr_lr[0] if len(curr_lr) == 1 else curr_lr
            print(f"Epoch {epoch}, batch {i}, loss={loss:.3f}, lr={lr_to_show}")

        lr_backbone_history.append(curr_lr[0])
        lr_head_history.append(curr_lr[-1])

        print("= = = = = = = = = =")
        if calculate_epoch_loss or save_best_model:
            last_lr = optimizer.param_groups[-1]["lr"]
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

        if not pretrain:
            if scheduler_type == "reduce_on_plato":
                scheduler.step(check_loss_value)
            else:
                scheduler.step()
        epoch += 1

    writer.close()

    if calculate_epoch_loss:
        tl = torch.Tensor(val_loss_history)
        best_idx = torch.argmin(tl).item()
        best_val = tl[best_idx].item()
        print(f"Best validation loss = {best_val} was reached at {best_idx+1} epoch.")

    save_model(model, run_folder, tag, backbone_name)

    if model_conf["weights_path"]:
        save_model(
            model,
            extend_relative_path_to_parent(model_conf["weights_path"]),
            tag,
            backbone_name,
        )

    loss_df = pd.DataFrame(
        {
            "epoch": range(1, epoch + 1),
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "best_val_loss": best_val_loss_history,
            "lr_head": lr_head_history,
            "lr_backbone": lr_backbone_history,
        }
    )
    loss_df.to_csv("losses.csv", index=False)


def main(config_path: str = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to config file")
    args = parser.parse_args()

    filepath = args.config or config_path
    train(filepath)


if __name__ == "__main__":
    main("config_example_quick_train_with_epoch_loss.json")
