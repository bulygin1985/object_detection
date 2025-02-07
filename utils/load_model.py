import os
from typing import Type

import torch
import torch.nn as nn
from config import load_config

from models.centernet import ModelBuilder


def create_model(
    model_conf: dict, data_conf: dict, device: torch.device, load_weights: bool = True
):
    model = ModelBuilder(
        filters_size=model_conf["head"]["filters_size"],
        alpha=model_conf["alpha"],
        class_number=data_conf.get("class_amount"),
        backbone=model_conf["backbone"]["name"],
        backbone_weights=(
            model_conf["backbone"]["pretrained_weights"] if load_weights else None
        ),
    )
    return model.to(device)


def create_model_from_config_file(
    config_filepath: str, device: torch.device, load_weights: bool = True
):
    """Creates a model given config."""
    model_conf, _, data_conf = load_config(config_filepath)
    return create_model(model_conf, data_conf, device, load_weights=load_weights)


def load_model(
    device: torch.device, config_filepath: str = None, checkpoint_path: str = None
) -> nn.Module:
    """
    Loads a PyTorch model from a checkpoint file and moves it to the specified device.

    Args:
        device (torch.device): The device (CPU or GPU) where the model will be loaded.
        config_filepath (str): path to config file.
        checkpoint_path (str, optional): Path to the model checkpoint file. Defaults to
            "../models/checkpoints/pretrained_weights.pt" if not provided.

    Returns:
        nn.Module: The loaded model instance.

    Raises:
        FileNotFoundError: If the specified checkpoint file does not exist.
    """
    checkpoint_path = (
        "../models/checkpoints/pretrained_weights.pt"
        if checkpoint_path is None
        else checkpoint_path
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = create_model_from_config_file(config_filepath, device, False)
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    return model
