import argparse
from os import path

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils import data

from data.dataset import Dataset
from models.centernet import ModelBuilder, input_height, input_width
from training.encoder import CenternetEncoder

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overfit", action="store_true", help="overfit to 10 images")
parser.add_argument(
    "-b", "--backbone", type=str, help="Backbone name (empty, default, resnetXX,)"
)
parser.add_argument(
    "-bw",
    "--backbone_weights",
    type=str,
    help="Backbone weights for supported pretrained backbones",
)
args = parser.parse_args()

overfit = args.overfit

dataset_val = torchvision.datasets.VOCDetection(
    root="../VOC", year="2007", image_set="train", download=False
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

if overfit:
    tag = "overfit"
    training_data = torch.utils.data.Subset(torch_dataset, range(10))
    lr = 0.05
    batch_size = 10
    min_lr = 2e-3
    patience = 100
    stop_loss = 1.0
    stop_epoch = None
else:
    tag = ""
    min_lr = 1e-5
    patience = 7
    stop_loss = None
    stop_epoch = 500


def criteria_satisfied(current_loss, current_epoch):
    if stop_loss is not None and current_loss < 1.0:
        return True
    if stop_epoch is not None and current_epoch > stop_epoch:
        return True
    return False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 0.25
if args.backbone and args.backbone != "default":
    alpha = 1.0
model = ModelBuilder(
    alpha=alpha, backbone=args.backbone, backbone_weights=args.backbone_weights
)
model = model.to(device)

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
    training_data, num_workers=4, batch_size=batch_size, shuffle=True
)

epoch = 1
get_desired_loss = False

while True:
    print("EPOCH {}:".format(epoch))

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
        print(
            f"Epoch {epoch}, batch {i}, loss={loss:.3f}, lr={scheduler.get_last_lr()}"
        )

    if criteria_satisfied(loss, epoch):
        break

    scheduler.step(loss_dict["loss"])

    epoch += 1

train_location = path.dirname(path.abspath(__file__))
checkpoints_dir = path.join(train_location, "../models/checkpoints")
tail = f"_{tag}" if tag else ""
checkpoint_filename = path.join(checkpoints_dir, f"pretrained_weights{tail}.pt")
torch.save(model.state_dict(), checkpoint_filename)
