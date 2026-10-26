from ablation.configs._base import make_config


CONFIG = make_config(
    name="cifar100_wrn28_10_ablation_local_sensitivity_only",
    output_dir=(
        'path/to/output/CIFAR100_WRN28-10_local_sensitivity_only_sleep20_weight0p05'
    ),
    variant="local_sensitivity_only",
)

