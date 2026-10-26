# CIFAR-100 five-seed suite

This suite runs the four paper methods (baseline, exact GMR, Wake–Sleep
$$T=1$$, and Wake–Sleep $$T=20$$) for seeds 42--46. The evaluation-noise seed
is fixed at 42 for every training seed. Completed runs are skipped and an
incomplete run with `lastest.pt` is resumed automatically.

Run it from the project root with:

```powershell
python -B -u experiments/cifar100_five_seed/run_cifar100_suite.py
```

The runner uses the canonical configs in `configs/` and writes each run
directly to its descriptive `Results/CIFAR100_..._seedXX/` directory. Summary
files are written to `Results/cifar100_five_seed_summary/`.

Each method has one configuration file. The launcher supplies `--seed` and
`--output-dir` to `train.py`; no seed-specific configuration copies are needed.
