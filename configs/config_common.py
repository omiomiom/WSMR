"""Shared experiment defaults.

Dataset, model, method, and output settings are defined in ``config_*.py``.
"""

from __future__ import annotations


COMMON_CONFIG = {
    "experiment": {
        "seed": 42,
        "device": "auto",
        "deterministic": True,
    },
    "loader": {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    },
    "training": {
        "optimizer": "sgd",
        "momentum": 0.0,
        "weight_decay": 0.0,
        "gradient_clip_norm": None,
        "progress_update_interval": 20,
    },
    "regularizer": {
        # The objective uses (weight / 2) * regularizer.
        "weight": 0.05,
        "sigma": 0.15,
        "z_value": 1.5,
        "target_margin": 0.0,
        "softplus_beta": 5.0,
        "variance_epsilon": 1.0e-8,
        # activation_rms weights squared gradients by channel activation second moments.
        # This compensates for equivalent channel rescaling when clipping is inactive.
        "hidden_gradient_metric": "euclidean",  # euclidean | activation_rms
        "hidden_metric_epsilon": 1.0e-6,
    },
    "wake_sleep": {
        "factor_storage_dtype": "float32",
        "factor_contraction_dtype": "float32",
    },
    "evaluation": {
        "every_epochs": 1,
        "noise_type": "gaussian",
        "noise_sigma": 0.15,
        "noise_clip": True,
        "noise_seed": 42,
        "foreground_flip_probability": 0.45,
        "background_flip_probability": 0.15,
        "binary_threshold": 0.5,
        "bn_calibration_batches": 50,
    },
}
