from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from observability.anomaly import zscore_detector
from observability.distribution import detect_distribution_shift


def approximate_token_lengths(texts: Iterable[str] | Any) -> list[int]:
    if texts is None:
        return []
    return [len(str(t).split()) for t in texts if t is not None]


def detect_text_length_shift(
    current_texts: Iterable[str] | Any,
    baseline_batch_means: Iterable[float] | Any,
    threshold: float = 3.0,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    lengths = approximate_token_lengths(current_texts)
    current_mean = float(np.mean(lengths)) if lengths else 0.0
    clean_baseline = [float(x) for x in baseline_batch_means if x is not None and np.isfinite(float(x))]
    result = zscore_detector(current_mean, clean_baseline, threshold=threshold, *args, **kwargs)
    result["metric"] = "mean_text_length"
    result["current_mean"] = current_mean
    return result


def detect_embedding_norm_shift(
    current_norms: Iterable[float] | Any,
    baseline_norms: Iterable[float] | Any,
    threshold: float = 3.0,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Use norm-distribution drift as a model-free embedding health signal."""
    result = detect_distribution_shift(current_norms, baseline_norms, ratio_threshold=threshold, *args, **kwargs)
    result["metric"] = "embedding_norm_distribution"
    result["method"] = f"embedding:{result['method']}"
    return result
