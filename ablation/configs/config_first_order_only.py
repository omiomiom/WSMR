from ablation.configs._base import make_config


CONFIG = make_config(
    name="cifar100_wrn28_10_ablation_first_order_only",
    output_dir='path/to/output/CIFAR100_WRN28-10_first_order_only_weight0p05',
    variant="first_order_only",
)

