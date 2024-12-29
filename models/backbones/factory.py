from .centernet_backbone import Backbone
from .efficientnet import create_efficientnet_backbone
from .mobilenetv2 import create_mobilenetv2_backbone
from .resnet import create_resnet_backbone


def create_backbone(backbonename: str, alpha: float, weights: str = None):
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
