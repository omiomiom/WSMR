from copy import deepcopy

from configs.config_common import COMMON_CONFIG


CONFIG = deepcopy(COMMON_CONFIG)
CONFIG["dataset"] = "imagenet1k"
CONFIG["model"] = "resnet18"
CONFIG["experiment"]["name"] = "imagenet1k_resnet18_exact_g0"
CONFIG["experiment"]["method"] = "exact_g0"
CONFIG["paths"] = {
    "dataset_root": 'path/to/data',
    "output_dir": (
        'path/to/output/ImageNet1K_ResNet18_exact_g0_beta5_z1p5_weight0p05'
    ),
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
CONFIG["regularizer"]["weight"] = 0.05
CONFIG["regularizer"]["softplus_beta"] = 5.0
CONFIG["regularizer"]["z_value"] = 1.5
