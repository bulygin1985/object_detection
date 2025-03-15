import torch


def split_params_for_weight_decay(model, decay_bias):
    """splits model named parameters into 'regular' and 'batch_norm' group."""
    names_all = set(model.state_dict().keys())
    bn_param_tail = ".running_var"

    suffix_to_check = [".weight"]
    if decay_bias:
        suffix_to_check.append(".bias")

    def should_decay_name(name):
        if not decay_bias and name.endswith(".bias"):
            return False
        for suffix in suffix_to_check:
            if (
                name.endswith(suffix)
                and (name[: -len(suffix)] + bn_param_tail) in names_all
            ):
                return False
        return True

    result_named_params = list(
        [(name, p) for name, p in model.named_parameters() if should_decay_name(name)]
    )
    result_named_nodecay_params = list(
        [
            (name, p)
            for name, p in model.named_parameters()
            if not should_decay_name(name)
        ]
    )
    return result_named_params, result_named_nodecay_params


def optimizer_type_by_str(name: str):
    if name == "Adam":
        return torch.optim.Adam
    elif name == "AdamW":
        return torch.optim.AdamW
    elif name == "SGD":
        return torch.optim.SGD
    raise ValueError(f"optimizer '{name}' is not supported.")


def name_fits(name, include_patterns=None, exclude_patterns=None):
    """Check if name has any of include patterns and does not have all exclude patterns."""
    if include_patterns and not any([p in name for p in include_patterns]):
        return False
    if exclude_patterns and not all([p not in name for p in exclude_patterns]):
        return False
    return True


def filter_named_values_by_pattern(
    named_values, include_patterns=None, exclude_patterns=None
):
    """Filter sequence (name, value) for name to have one of prefixes and none of exclude prefixes."""
    return [
        p
        for name, p in named_values
        if name_fits(name, include_patterns, exclude_patterns)
    ]


def create_scheduler(optimizer, scheduler_conf):
    scheduler_type = scheduler_conf["type"]
    conf = scheduler_conf[scheduler_type]
    if scheduler_type == "reduce_on_plato":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=conf["factor"],
            patience=conf["patience"],
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=1,
            min_lr=conf["min_lr"],
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=conf["step_size"],
            gamma=conf["factor"],
            last_epoch=conf.get("last_epoch", -1),
        )
    elif scheduler_type == "multi_step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=conf["milestones"],
            gamma=conf["factor"],
            last_epoch=conf.get("last_epoch", -1),
        )
    raise RuntimeError(f"Unsupported learning rate scheduler type '{scheduler_type}'.")
