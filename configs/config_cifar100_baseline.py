from copy import deepcopy

from configs.config_common import COMMON_CONFIG


CONFIG = deepcopy(COMMON_CONFIG)
CONFIG["dataset"] = "cifar100"
CONFIG["model"] = "wrn28_10"
CONFIG["experiment"]["name"] = "cifar100_wrn28_10_baseline_seed42"
CONFIG["experiment"]["method"] = "baseline"
CONFIG["paths"] = {
    "dataset_root": 'path/to/data',
    "output_dir": 'path/to/output/CIFAR100_WRN28-10_baseline_seed42',
}
CONFIG["dataset_options"] = {
    "download": True,
    "train_augmentation": False,
}
CONFIG["loader"]["batch_size"] = 128
CONFIG["loader"]["eval_batch_size"] = 256
CONFIG["model_options"] = {
    "dropout": 0.0,
    "zero_init_residual": False,
}
CONFIG["training"]["epochs"] = 100
CONFIG["training"]["learning_rate"] = 0.1
CONFIG["training"]["lr_schedule"] = {"61": 0.02, "81": 0.004}
CONFIG["training"]["checkpoint_every"] = 10
# Inactive for baseline; records the regularizer settings used by the comparison methods.
CONFIG["regularizer"]["weight"] = 0.05
