# WSMR: Wake–Sleep Margin Regularization

PyTorch implementation of the baseline, Gaussian Margin Regularization (GMR), and Wake–Sleep Margin Regularization (WSMR) experiments reported in the paper. The repository includes training, perturbation evaluation, and core ablations for MNIST, CIFAR-100, and ImageNet-1K.

## Installation and project structure

Use Python 3.10 or later and a PyTorch/CUDA installation suitable for your device:

```bash
python -m pip install -r requirements.txt
```

Run all commands from the repository root.

```text
configs/                         One base configuration per experiment
wslgmr/                          Models, data loading, training, and regularization
experiments/                     Serial experiment launchers and summaries
ablation/                        Core ablation configurations and launchers
tests/                           Project layout checks
train.py                         Single-run training entry point
reevaluate.py                    Checkpoint reevaluation
lmdn_attack_protocol.py          FGSM, IFGSM, and transfer-attack evaluation
```

Datasets, training results, checkpoints, and local logs are not included. Generated files should be stored outside the repository or under the Git-ignored `data/` and `Results/` directories.

## Data preparation and configuration paths

All data and output paths in the configuration files are placeholders. Replace them with your own paths before running an experiment:

```python
CONFIG["paths"] = {
    "dataset_root": "path/to/data",
    "output_dir": "path/to/output/experiment_name",
}
```

Use a separate output directory for each experiment. The `--output-dir` option can override the output location at launch time.

MNIST and CIFAR-100 are downloaded automatically to `dataset_root` when `dataset_options.download` is `True`, as enabled in the supplied configurations.

Download ImageNet-1K from the [official ImageNet download page](https://www.image-net.org/download.php) and follow its access terms. Prepare the training and validation splits as class-specific directories under the configured root:

```text
path/to/data/
├── train/<class folders>
└── val/<class folders>
```

ImageNet evaluation uses the dataset path in the corresponding training configuration.

## Training experiments

### MNIST / MLP

```bash
python -B -u train.py configs/config_mnist_baseline.py
python -B -u train.py configs/config_mnist_exact_g0.py
python -B -u train.py configs/config_mnist_gi_i1.py
python -B -u train.py configs/config_mnist_gi_i20.py
```

### CIFAR-100 / WideResNet-28-10

```bash
python -B -u train.py configs/config_cifar100_baseline.py
python -B -u train.py configs/config_cifar100_exact_g0.py
python -B -u train.py configs/config_cifar100_gi_i1.py
python -B -u train.py configs/config_cifar100_gi_i20.py
```

### ImageNet-1K / ResNet-18

```bash
python -B -u train.py configs/config_imagenet1k_resnet18_baseline.py
python -B -u train.py configs/config_imagenet1k_resnet18_exact_g0.py
python -B -u train.py configs/config_imagenet1k_resnet18_gi_i1.py
python -B -u train.py configs/config_imagenet1k_resnet18_gi_i20.py
```

For each group above, the four commands correspond to the baseline, GMR, WSMR T=1, and WSMR T=20, respectively.

### ImageNet-1K / ResNet-50

The repository provides the baseline and WSMR T=20 configurations used in the paper:

```bash
python -B -u train.py configs/config_imagenet1k_baseline.py
python -B -u train.py configs/config_imagenet1k_gi_i20.py
```

### Command-line options

Use `--seed` to change the training seed without duplicating a configuration file:

```bash
python -B -u train.py configs/config_cifar100_gi_i20.py --seed 43
```

Override the result directory with:

```bash
python -B -u train.py configs/config_cifar100_gi_i20.py --output-dir path/to/output/my_run
```

A result directory contains `resolved_config.json`, `status.json`, `history.json`, `history.csv`, `best.pt`, `lastest.pt`, and periodic checkpoints. The filename `lastest.pt` is retained by the implementation.

Use `--resume` to continue from `lastest.pt`, or specify a saved epoch:

```bash
python -B -u train.py configs/config_cifar100_gi_i20.py --resume
python -B -u train.py configs/config_cifar100_gi_i20.py --resume 50
```

Print the resolved configuration without starting training:

```bash
python -B train.py configs/config_cifar100_gi_i20.py --seed 43 --dry-run
```

## Core ablations

The five CIFAR-100 ablations are exact delayed accumulation, T=5, T=40, the first-order branch only, and the local-sensitivity branch only. Each setting has one configuration file and is repeated over five seeds by the launcher:

```bash
python -B -u ablation/repeats/run_core_suite.py
```

Run one setting, for example exact delayed accumulation:

```bash
python -B -u ablation/repeats/run_five_seed.py exact_delayed
```

See [ablation/README.md](ablation/README.md) for the available settings and single-run commands. After completing the formal CIFAR-100 runs and ablations, audit the results and generate summaries and curves:

```bash
python -B ablation/repeats/summarize_five_seed.py
```

Repeat launchers and summaries use experiment-specific directories under `Results/`. Set the data path in the base configurations before launching these workflows. When combining single runs with a suite or summary, use the output-directory layout documented in the corresponding launcher README.

## Target-margin experiment

Compare tau=1, 2, and 4 serially on CIFAR-100 with WSMR T=20, lambda=0.05, and training seed 42:

```bash
python -B -u experiments/run_cifar100_tau_sweep.py
```

The launcher uses the completed tau=0 formal run as its reference. Run `config_cifar100_gi_i20.py` first and configure the tau-specific output paths before launching the comparison. Different tau values represent distinct experiments rather than seed-specific configuration copies.

## Adversarial evaluation

Complete the corresponding training runs and set each configuration's `output_dir` to the directory containing its final checkpoint. The following commands evaluate white-box FGSM, white-box IFGSM, and transfer attacks generated by the baseline model:

```bash
python -B -u lmdn_attack_protocol.py --family mnist_mlp
python -B -u lmdn_attack_protocol.py --family cifar100_wrn28_10
python -B -u lmdn_attack_protocol.py --family imagenet1k_resnet18 --family imagenet1k_resnet50 --sample-count 5000 --output-dir path/to/output/imagenet_attacks
```

MNIST and CIFAR-100 use their full test sets. The ImageNet command uses a fixed, class-balanced validation subset of 5,000 images. Repeating a command resumes unfinished perturbation radii. The evaluator checks sample and checkpoint hashes.

Plot the attack results:

```bash
python -B plot_lmdn_attack_protocol.py --input-dir path/to/output/imagenet_attacks
```

Reevaluate the final checkpoint of one training run:

```bash
python -B -u reevaluate.py configs/config_cifar100_gi_i20.py --checkpoint lastest.pt
```

## Configuration and method details

The supplied configurations retain the existing experiment hyperparameters: 20 epochs for MNIST, 100 for CIFAR-100, and 90 for ImageNet-1K, with a GMR/WSMR regularization weight of 0.05. Each training run writes its resolved configuration to `resolved_config.json`.

For the method formulation and derivations, see Section 3 of the paper. The [CIFAR-100 suite README](experiments/cifar100_five_seed/README.md) describes the repeated-experiment workflow.
