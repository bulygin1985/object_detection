import argparse
import time
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


def split_params_for_weight_decay(model, decay_bias):
    """splits model named parameters into 'regular' and 'batch_norm' group."""
    names_all = set(model.state_dict().keys())
    result = []
    bn_param_tail = ".running_var"

    suffix_to_check = [".weight"]
    if decay_bias:
        suffix_to_check.append(".bias")

    def should_decay_name(name):
        if not decay_bias and name.endswith(".bias"):
            return False
        for suffix in suffix_to_check:
            if (
                name.endswith(suffix)
                and (name[: -len(suffix)] + bn_param_tail) in names_all
            ):
                return False
        return True

    result_named_params = list(
        [(name, p) for name, p in model.named_parameters() if should_decay_name(name)]
    )
    result_named_nodecay_params = list(
        [
            (name, p)
            for name, p in model.named_parameters()
            if not should_decay_name(name)
        ]
    )
    return result_named_params, result_named_nodecay_params


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


def compose_transforms(data_augmentation_params=None):
    transforms_list = [transforms.Resize(size=(IMG_WIDTH, IMG_HEIGHT))]
    if data_augmentation_params:
        resize_crop_scale = data_augmentation_params.get(
            "random_resize_crop_scale", None
        )
        resize_crop_ratio = data_augmentation_params.get(
            "random_resize_crop_ratio", None
        )
        if resize_crop_scale is not None:
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


def calculate_loss_batch_generator(model, batch_generator):
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


def calculate_loss(model, data, batch_size=32, num_workers=0):
    batch_generator = torch.utils.data.DataLoader(
        data,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return calculate_loss_batch_generator(model, batch_generator)


def name_fits(name, include_patterns=None, exclude_patterns=None):
    """Check if name has any of include prefixes and does not have all exclude prefixes."""
    if include_patterns and not any([p in name for p in include_patterns]):
        return False
    if exclude_patterns and not all([p not in name for p in exclude_patterns]):
        return False
    return True


def filter_named_values_by_prefix(
    named_values, include_prefixes=None, exclude_prefixes=None
):
    """Filter sequence (name, value) for name to have one of prefixes and none of exclude prefixes."""
    return [
        p
        for name, p in named_values
        if name_fits(name, include_prefixes, exclude_prefixes)
    ]


def train(model_conf, train_conf, data_conf):
    # torch.manual_seed(42)

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

    train_transform = compose_transforms(train_conf.get("data_augmentation"))
    val_transform = compose_transforms()

    encoder = CenternetEncoder(IMG_HEIGHT, IMG_WIDTH)

    dataset_val = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset_val)
    dataset_train = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset_train)
    val_data = Dataset(
        dataset=dataset_val, transformation=val_transform, encoder=encoder
    )
    train_data = Dataset(
        dataset=dataset_train, transformation=train_transform, encoder=encoder
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
        imagenet_normalization=model_conf.get("imagenet_normalization", False),
        conv_bias=model_conf["head"].get("conv_bias", False),
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
        trainable_backbone_params = filter_named_values_by_prefix(
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

    def create_scheduler(optimizer, scheduler_conf):
        scheduler_type = lr_schedule_conf["type"]
        conf = lr_schedule_conf[scheduler_type]
        if scheduler_type == "reduce_on_plato":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=conf["factor"],
                patience=conf["patience"],
                threshold=1e-4,
                threshold_mode="rel",
                cooldown=1,
                min_lr=conf["min_lr"],
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=conf["step_size"],
                gamma=conf["factor"],
                last_epoch=conf.get("last_epoch", -1),
            )
        elif scheduler_type == "multi_step":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=conf["milestones"],
                gamma=conf["factor"],
                last_epoch=conf.get("last_epoch", -1),
            )
        raise RuntimeError(
            f"Unsupported learning rate scheduler type '{scheduler_type}'."
        )

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
            backbone_decay_params = filter_named_values_by_prefix(
                backbone_decay_params,
                bb_train_params_patterns_include,
                bb_train_params_patterns_exclude,
            )
            backbone_nodecay_params = filter_named_values_by_prefix(
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
    optimizer = torch.optim.Adam(opt_params, lr=0.0)

    model.train(True)
    persistent_workers = train_conf.get("persistent_workers", False)

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
    best_val_loss = 1e9

    calculate_epoch_loss = train_conf.get("calculate_epoch_loss")
    save_best_model = train_conf.get("save_best_model", True)
    skip_save_best_model_epochs = train_conf.get("skip_save_best_model_epochs", 0)

    num_workers_validation = train_conf.get("num_workers_validation", num_workers)
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

        print(f"= = = = = = = = = =")
        if calculate_epoch_loss:
            train_loss_history.append(
                calculate_loss_batch_generator(model, batch_generator_train_val)
            )
            val_loss_history.append(
                calculate_loss_batch_generator(model, batch_generator_val)
            )
            if val_loss_history[-1] < best_val_loss:
                best_val_loss = val_loss_history[-1]
                if save_best_model and epoch > skip_save_best_model_epochs:
                    save_model(
                        model,
                        model_conf["weights_path"],
                        tag=tag + "_best",
                        backbone=model_conf["backbone"]["name"],
                    )

            print(
                f"Epoch {epoch} train loss = {train_loss_history[-1]:.5f}, val loss = {val_loss_history[-1]:.5f}, best val loss = {best_val_loss:.5f}"
            )

            best_val_loss_history.append(best_val_loss)

        print(
            f"Epoch {epoch} calculation time is {time.perf_counter()-epoch_start} seconds"
        )
        print(f"= = = = = = = = = =")
        if criteria_satisfied(loss, epoch):
            break

        check_loss_value = train_loss_history[-1] if calculate_epoch_loss else loss

        if not pretrain:
            if scheduler_type == "reduce_on_plato":
                scheduler.step(check_loss_value)
            else:
                scheduler.step()
        epoch += 1

    if calculate_epoch_loss:
        tl = torch.Tensor(val_loss_history)
        best_idx = torch.argmin(tl).item()
        best_val = tl[best_idx].item()
        print(f"Best validation loss = {best_val} was reached at {best_idx+1} epoch.")

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

    save_model(
        model,
        model_conf["weights_path"],
        tag=tag,
        backbone=model_conf["backbone"]["name"],
    )


if __name__ == "__main__":
    main()
