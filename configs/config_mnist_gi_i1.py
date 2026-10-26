from copy import deepcopy

from configs.config_common import COMMON_CONFIG


CONFIG = deepcopy(COMMON_CONFIG)
CONFIG["dataset"] = "mnist"
CONFIG["model"] = "mlp"
CONFIG["experiment"]["name"] = "mnist_mlp_wake_sleep_t1_seed42"
CONFIG["experiment"]["method"] = "replace_gi"
CONFIG["paths"] = {
    "dataset_root": 'path/to/data',
    "output_dir": (
        'path/to/output/MNIST_MLP_wake_sleep_t1_beta5_z1p5_weight0p05_seed42'
    ),
}
CONFIG["dataset_options"] = {
    "download": True,
    "train_augmentation": False,
}
CONFIG["loader"]["batch_size"] = 128
CONFIG["loader"]["eval_batch_size"] = 256
CONFIG["model_options"] = {
    "dropout": 0.0,
    "mlp_layers": [784, 256, 256, 10],
    "zero_init_residual": False,
}
CONFIG["training"]["epochs"] = 20
CONFIG["training"]["learning_rate"] = 0.1
CONFIG["training"]["lr_schedule"] = {}
CONFIG["training"]["checkpoint_every"] = 5
CONFIG["regularizer"]["weight"] = 0.05
CONFIG["regularizer"]["softplus_beta"] = 5.0
CONFIG["regularizer"]["z_value"] = 1.5
CONFIG["regularizer"]["target_margin"] = 4.0
CONFIG["regularizer"]["hidden_gradient_metric"] = "activation_rms"
CONFIG["wake_sleep"]["sleep_interval"] = 1
CONFIG["evaluation"]["noise_type"] = "binary_flip"
