import argparse

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils import data

from data.dataset import Dataset
from models.centernet import ModelBuilder, input_height, input_width
from training.encoder import CenternetEncoder

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overfit", action="store_true", help="overfit to 10 images")
args = parser.parse_args()

overfit = args.overfit

image_set = "train"

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


def criteria_satisfied(_, current_epoch):
    if current_epoch >= 10000:
        return True
    return False


if overfit:
    subset_len = 10
    training_data = torch.utils.data.Subset(torch_dataset, range(subset_len))
    batch_size = subset_len
    patience = 50
    image_set = "val"

    def criteria_satisfied(current_loss, _):
        if current_loss < 100.0:
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
    factor=0.1,
    patience=patience,
    threshold=1e-4,
    threshold_mode="rel",
    cooldown=1,
    min_lr=1e-2,
)

model.train(True)

batch_generator = torch.utils.data.DataLoader(
    training_data, num_workers=0, batch_size=batch_size, shuffle=False
)

epoch = 1
get_desired_loss = False

while True:
    print("EPOCH {}:".format(epoch))

    loss_dict = {}
    for _, data in enumerate(batch_generator):
        input_data, gt_data = data
        input_data = input_data.to(device).contiguous()

        gt_data = gt_data.to(device)
        gt_data.requires_grad = False

        loss_dict = model(input_data, gt=gt_data)
        optimizer.zero_grad()  # compute gradient and do optimize step
        loss_dict["loss"].backward()

        optimizer.step()
        loss = loss_dict["loss"]
        print(f" loss={loss}, lr={scheduler.get_last_lr()}")

    if criteria_satisfied(loss_dict["loss"], epoch):
        break

    scheduler.step(loss_dict["loss"])

    epoch += 1

torch.save(model.state_dict(), "../models/checkpoints/pretrained_weights.pt")
