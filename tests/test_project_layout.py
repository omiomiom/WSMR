from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PaperProjectLayoutTests(unittest.TestCase):
    def test_documented_entry_points_exist(self) -> None:
        paths = (
            "train.py",
            "experiments/cifar100_five_seed/run_cifar100_suite.py",
            "ablation/repeats/run_core_suite.py",
            "ablation/repeats/run_five_seed.py",
            "ablation/repeats/summarize_five_seed.py",
            "lmdn_attack_protocol.py",
        )
        for relative in paths:
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_paper_configs_exist(self) -> None:
        configs = (
            "config_mnist_baseline.py",
            "config_mnist_exact_g0.py",
            "config_mnist_gi_i1.py",
            "config_mnist_gi_i20.py",
            "config_cifar100_baseline.py",
            "config_cifar100_exact_g0.py",
            "config_cifar100_gi_i1.py",
            "config_cifar100_gi_i20.py",
            "config_imagenet1k_resnet18_baseline.py",
            "config_imagenet1k_resnet18_exact_g0.py",
            "config_imagenet1k_resnet18_gi_i1.py",
            "config_imagenet1k_resnet18_gi_i20.py",
            "config_imagenet1k_baseline.py",
            "config_imagenet1k_gi_i20.py",
        )
        for name in configs:
            with self.subTest(config=name):
                self.assertTrue((ROOT / "configs" / name).is_file())


if __name__ == "__main__":
    unittest.main()
