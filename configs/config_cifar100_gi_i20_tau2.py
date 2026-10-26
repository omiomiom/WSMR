from copy import deepcopy
from configs.config_cifar100_gi_i20 import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["regularizer"]["target_margin"] = 2.0
CONFIG["experiment"]["name"] = "cifar100_wrn28_10_wake_sleep_t20_tau2_seed42"
CONFIG["paths"]["output_dir"] = 'path/to/output/CIFAR100_WRN28-10_wake_sleep_t20_tau2_beta5_z1p5_weight0p05_seed42'
