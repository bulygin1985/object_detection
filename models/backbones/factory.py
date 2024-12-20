from .centernet_backbone import Backbone


def create_bakbone(backbonename: str, alpha: float):
    if not backbonename or backbonename == "default":
        return Backbone(alpha)
    raise ValueError(f"Backbone {backbonename} is not supported yet.")
