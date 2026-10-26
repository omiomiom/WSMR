# Reproducibility workflows

This directory contains standalone experiment orchestration rather than model
or method implementation. Each subdirectory owns the launcher and summarizer
for one reproducibility workflow while reusing the canonical configs in the
project-level `configs/` directory.

- `cifar100_five_seed/`: the five-random-seed comparison of baseline, exact
  GMR, Wake--Sleep $$T=1$$ and Wake--Sleep $$T=20$$ on CIFAR-100.

Method-specific ablations remain in the top-level `ablation/` directory, so
they stay visibly separate from both the core package and reproduction work.
