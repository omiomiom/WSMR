# CIFAR-100 paper ablations

This directory isolates the five mechanism ablations reported in the paper
from the production training path:

1. `exact_delayed`: accumulate the exact parameter-shaped local update for
   20 batches before applying it;
2. `sleep5`: use the complete method with sleep interval $$T=5$$;
3. `sleep40`: use the complete method with sleep interval $$T=40$$;
4. `first_order_only`: retain cross-entropy and the probability-weighted
   first-order margin branch only;
5. `local_sensitivity_only`: retain cross-entropy and the delayed local
   sensitivity branch only.

Every setting uses CIFAR-100, WideResNet-28-10, 100 epochs, seeds 42--46,
regularization weight 0.05, $$\beta=5$$, and $$z=1.5$$. The validation-noise
seed remains fixed at 42.

## One-command reproduction

Run all five paper ablations, strictly serially:

```powershell
python -B -u ablation/repeats/run_core_suite.py
```

Run one five-seed ablation:

```powershell
python -B -u ablation/repeats/run_five_seed.py exact_delayed
python -B -u ablation/repeats/run_five_seed.py sleep5
python -B -u ablation/repeats/run_five_seed.py sleep40
python -B -u ablation/repeats/run_five_seed.py first_order_only
python -B -u ablation/repeats/run_five_seed.py local_sensitivity_only
```

The launchers skip completed seeds and resume an incomplete seed when its
`lastest.pt` exists. Results are stored under
`Results/ablation/repeats/<variant>/`.

After the formal CIFAR-100 runs and all five ablations are complete, audit,
summarize, and reproduce the paper figures with:

```powershell
python -B ablation/repeats/summarize_five_seed.py
```

Use `--paper-figures-dir path/to/figures` only when a second copy should be
placed in a manuscript repository.

The ablation entry point also accepts a specific config, seed, and output
directory for controlled runs:

```powershell
python -B -u ablation/train.py ablation/configs/config_exact_delayed.py --seed 42 --output-dir Results/ablation/repeats/exact_delayed/CIFAR100_WRN28-10_exact_delayed_seed42
```
