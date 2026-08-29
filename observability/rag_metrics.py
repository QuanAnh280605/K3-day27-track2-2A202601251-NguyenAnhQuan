from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from observability.anomaly import zscore_detector


def approximate_token_lengths(texts: Iterable[str]) -> list[int]:
    # Deliberately simple proxy; no tokenizer/model download needed.
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
    result = zscore_detector(current_mean, clean_baseline, threshold=threshold)
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
    """Detect embedding norm drift using z-score and statistical comparison against baseline."""
    clean_cur = [float(x) for x in current_norms if x is not None and np.isfinite(float(x))]
    clean_base = [float(x) for x in baseline_norms if x is not None and np.isfinite(float(x))]
    
    cur = np.asarray(clean_cur, dtype=float)
    base = np.asarray(clean_base, dtype=float)
    if cur.size == 0 or base.size == 0:
        return {"is_anomaly": False, "score": 0.0, "method": "embedding_norm_zscore", "reason": "empty_input"}
    
    cur_mean = float(np.mean(cur))
    base_mean = float(np.mean(base))
    base_std = float(np.std(base))
    
    if base_std == 0:
        score = float("inf") if cur_mean != base_mean else 0.0
    else:
        score = abs(cur_mean - base_mean) / base_std
        
    is_anomaly = bool(score > threshold)
    return {
        "is_anomaly": is_anomaly,
        "score": float(score),
        "method": "embedding_norm_zscore",
        "reason": f"current_norm_mean={cur_mean:.4f}, baseline_norm_mean={base_mean:.4f}, std={base_std:.4f}",
        "current_mean": cur_mean,
        "baseline_mean": base_mean,
    }


