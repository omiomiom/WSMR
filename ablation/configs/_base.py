from __future__ import annotations

from copy import deepcopy

from configs.config_cifar100_gi_i20 import CONFIG as CIFAR100_GI_I20_CONFIG


def make_config(
    *,
    name: str,
    output_dir: str,
    variant: str,
    sleep_interval: int = 20,
) -> dict:
    config = deepcopy(CIFAR100_GI_I20_CONFIG)
    config["experiment"]["name"] = name
    config["paths"]["output_dir"] = output_dir
    config["wake_sleep"]["sleep_interval"] = int(sleep_interval)
    config["wake_sleep"]["pairing_seed"] = int(config["experiment"]["seed"])
    config["ablation"] = {
        "variant": variant,
        "protocol": "cifar100_wrn28_10_seed42_100epochs",
    }
    return config
