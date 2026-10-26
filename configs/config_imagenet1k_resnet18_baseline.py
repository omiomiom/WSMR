from copy import deepcopy

from configs.config_common import COMMON_CONFIG


CONFIG = deepcopy(COMMON_CONFIG)
CONFIG["dataset"] = "imagenet1k"
CONFIG["model"] = "resnet18"
CONFIG["experiment"]["name"] = "imagenet1k_resnet18_baseline"
CONFIG["experiment"]["method"] = "baseline"
CONFIG["paths"] = {
    "dataset_root": 'path/to/data',
    "output_dir": 'path/to/output/ImageNet1K_ResNet18_baseline',
}
CONFIG["dataset_options"] = {
    "download": False,
    "download_devkit": True,
    "prepare_archives": True,
    "train_augmentation": True,
}
CONFIG["loader"]["batch_size"] = 128
CONFIG["loader"]["eval_batch_size"] = 128
CONFIG["model_options"] = {
    "dropout": 0.0,
    "zero_init_residual": False,
}
CONFIG["training"]["epochs"] = 90
CONFIG["training"]["learning_rate"] = 0.1
CONFIG["training"]["lr_schedule"] = {"31": 0.01, "61": 0.001}
CONFIG["training"]["checkpoint_every"] = 1
# Inactive for baseline; records the regularizer weight used by the ImageNet comparison methods.
CONFIG["regularizer"]["weight"] = 0.05
