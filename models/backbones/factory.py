from .centernet_backbone import Backbone


def create_bakbone(backbonename: str, alpha: float, weights: str = None):
    if not backbonename or backbonename == "default":
        assert weights is None
        return Backbone(alpha)
    raise ValueError(f"Backbone '{backbonename}' is not supported yet.")
