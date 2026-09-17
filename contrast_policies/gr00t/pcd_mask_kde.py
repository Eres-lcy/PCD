"""KDE policy-output selection for GR00T PCD-Mask."""

from __future__ import annotations

import numpy as np


def kde_density(
    samples: np.ndarray,
    reference: np.ndarray,
    bandwidth: np.ndarray,
) -> np.ndarray:
    standardized = (
        samples[:, :, None] - reference[:, None, :]
    ) / bandwidth[:, None, None]
    kernel = np.exp(-0.5 * standardized * standardized) / np.sqrt(2.0 * np.pi)
    return kernel.mean(axis=-1) / bandwidth[:, None]


def select_policy_output(
    samples: np.ndarray,
    contrast_samples: np.ndarray,
    *,
    alpha: float,
    bandwidth_factor: float,
    keep_threshold: float,
) -> np.ndarray:
    """Apply the same coordinate-wise KDE contrast rule as Pi0 PCD-Mask."""
    if samples.shape != contrast_samples.shape or samples.ndim != 3:
        raise ValueError(
            "PCD-Mask samples must have matching "
            "[num_repeats,horizon,action_dim] shapes"
        )
    repeats, horizon, action_dim = samples.shape
    if repeats < 2:
        raise ValueError("PCD-Mask num_repeats must be at least 2")

    data = samples.reshape(repeats, -1).T.astype(np.float64)
    contrast = contrast_samples.reshape(repeats, -1).T.astype(np.float64)
    scott = 1.06 * repeats ** (-0.2)
    bandwidth = np.maximum(
        bandwidth_factor * scott * data.std(axis=-1, ddof=1), 1e-6
    )
    contrast_bandwidth = np.maximum(
        bandwidth_factor * scott * contrast.std(axis=-1, ddof=1), 1e-6
    )
    probability = kde_density(data, data, bandwidth)
    contrast_probability = kde_density(data, contrast, contrast_bandwidth)
    eps = np.finfo(np.float64).tiny
    log_score = np.log(np.maximum(probability, eps)) + alpha * (
        np.log(np.maximum(probability, eps))
        - np.log(np.maximum(contrast_probability, eps))
    )
    valid = probability >= keep_threshold * probability.max(
        axis=-1, keepdims=True
    )
    log_score[~valid] = -np.inf
    baseline_index = probability.argmax(axis=-1)
    contrast_index = log_score.argmax(axis=-1)
    coordinate = np.arange(data.shape[0])
    decoded = data[coordinate, baseline_index].reshape(horizon, action_dim)
    contrasted = data[coordinate, contrast_index].reshape(horizon, action_dim)
    decoded[:, :6] = contrasted[:, :6]
    return decoded.astype(np.float32)
