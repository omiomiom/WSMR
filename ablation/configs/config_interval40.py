from ablation.configs._base import make_config


CONFIG = make_config(
    name="cifar100_wrn28_10_ablation_interval40",
    output_dir='path/to/output/CIFAR100_WRN28-10_wake_sleep40_weight0p05',
    variant="full",
    sleep_interval=40,
)
