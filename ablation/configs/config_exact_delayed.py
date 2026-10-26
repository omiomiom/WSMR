from ablation.configs._base import make_config


CONFIG = make_config(
    name="cifar100_wrn28_10_ablation_exact_delayed",
    output_dir=(
        'path/to/output/CIFAR100_WRN28-10_exact_delayed_sleep20_weight0p05'
    ),
    variant="exact_delayed",
)

