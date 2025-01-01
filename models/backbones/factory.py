from .abstract_backbone import AbstractBackbone
from .default_backbone import Backbone
from .efficientnet import create_efficientnet_backbone
from .mobilenetv2 import create_mobilenetv2_backbone
from .resnet import create_resnet_backbone


def create_backbone(
    backbonename: str, alpha: float = 1.0, weights: str = None
) -> AbstractBackbone:
    """Create backbone.
    Args:
        backbonename (str): name of the backbone,
        alpha (float): model scaling parameter (if backbone supports it, otherwise 1.),
        weights (str): name of pretrained weights for torchvision preptrained models ('DEFAULT' will work fine).
    Returns:
        AbstractBackbone: backbone model
    """
    if not backbonename or backbonename == "default":
        assert not weights
        return Backbone(alpha)
    if backbonename.startswith("resnet"):
        assert alpha == 1.0, f"only alpha=1 is supported for {backbonename}."
        return create_resnet_backbone(backbonename, weights)
    if backbonename.startswith("efficientnet"):
        assert alpha == 1.0, f"only alpha=1 is supported for {backbonename}."
        return create_efficientnet_backbone(backbonename, weights)
    if backbonename == "mobilenet_v2":
        assert alpha == 1.0, f"only alpha=1 is supported for {backbonename}."
        return create_mobilenetv2_backbone(backbonename, weights)

    raise ValueError(f"Backbone '{backbonename}' is not supported yet.")
