from typing import TypeVar

import torchvision.models as models

from .abstract_backbone import AbstractBackbone, TorchvisionBackbone
from .default_backbone import Backbone
from .resnet import ResnetBackbone

BACKBONE_BUILDER_CONF = {
    "resnet": ResnetBackbone,
    "efficientnet": TorchvisionBackbone,
    "mobilenet_v2": TorchvisionBackbone,
    "default": Backbone,
}

BackboneType = TypeVar("BackboneType", bound=AbstractBackbone)


def create_backbone(
    backbone_name: str = "default", alpha: float = 1.0, weights: str = None
) -> BackboneType:
    """Create backbone.
    Args:
        backbone_name (str): name of the backbone,
        alpha (float): model scaling parameter (if backbone supports it, otherwise 1),
        weights (str): name of pretrained weights for torchvision pretrained models ('default' will work fine).
    Returns:
        BackboneType: backbone model which implements AbstractBackbone class
    """
    backbone_name_parsed = backbone_name.lower()
    try:
        backbone_class = BACKBONE_BUILDER_CONF[backbone_name_parsed]
    except KeyError as exc:
        raise ValueError(
            f"Backbone '{backbone_name}' is not supported yet"
        ) from exc

    if backbone_name_parsed == "default":
        return backbone_class(alpha)

    print(f"WARNING! Only alpha=1 is supported for {backbone_name}")

    model = models.get_model(backbone_name_parsed, weights=weights)
    return backbone_class(model)
