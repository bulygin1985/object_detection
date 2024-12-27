from .centernet_backbone import Backbone
from .resnet import create_resnet_backbone


def create_bakbone(backbonename: str, alpha: float, weights: str = None):
    if not backbonename or backbonename == "default":
        assert not weights
        return Backbone(alpha)
    if backbonename.startswith("resnet"):
        assert alpha == 1.0, f"only alpha=1 is supported at the moment {backbonename}."
        return create_resnet_backbone(backbonename, weights)
    raise ValueError(f"Backbone '{backbonename}' is not supported yet.")
