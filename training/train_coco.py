import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms

from data.dataset import Dataset
from data.dataset_loaders import MSCOCODatasetLoader
from models.centernet import ModelBuilder
from training.encoder import CenternetEncoder
from utils.config import IMG_HEIGHT, IMG_WIDTH, load_config


def criteria_builder(stop_loss, stop_epoch):
    def criteria_satisfied(current_loss, current_epoch):
        if stop_loss is not None and current_loss < stop_loss:
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
    print(f"Selected image_set: {data_conf["image_set"]}")

    dataset_loader = MSCOCODatasetLoader(
        data_conf["images_folder"], data_conf["ann_file"]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(size=(IMG_WIDTH, IMG_HEIGHT)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    encoder = CenternetEncoder(
        IMG_HEIGHT, IMG_WIDTH, n_classes=data_conf["class_amount"]
    )

    torch_dataset = Dataset(
        dataset=dataset_loader.get_dataset(),
        transformation=transform,
        encoder=encoder,
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
        filters_size=model_conf["head"]["filters_size"],
        alpha=model_conf["alpha"],
        class_number=data_conf["class_amount"],
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
    main("config_example.json")
