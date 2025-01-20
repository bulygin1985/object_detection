import os
from typing import Type

import torch
import torch.nn as nn


def load_model(
    device: torch.device,
    model_type: Type[nn.Module],
    checkpoint_path: str = None,
    alpha: float = 0.25,
):
    checkpoint_path = (
        "../models/checkpoints/pretrained_weights.pt"
        if checkpoint_path is None
        else checkpoint_path
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = model_type(filters_size=[128, 64, 32], alpha=alpha).to(device)
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    return model
