from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import stats


def detect_distribution_shift(
    current_values: Iterable[float] | Any,
    baseline_values: Iterable[float] | Any,
    ratio_threshold: float = 3.0,
    p_value_threshold: float = 0.05,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect distribution drift using Kolmogorov-Smirnov (KS) test combined with mean ratio.
    
    Catches shape, scale, and location shifts in continuous metrics.
    """
    clean_cur = [float(x) for x in current_values if x is not None and np.isfinite(float(x))]
    clean_base = [float(x) for x in baseline_values if x is not None and np.isfinite(float(x))]
    
    cur = np.asarray(clean_cur, dtype=float)
    base = np.asarray(clean_base, dtype=float)
    if cur.size == 0 or base.size == 0:
        return {"is_anomaly": False, "score": 0.0, "method": "ks_and_ratio", "reason": "empty_input"}
    
    cur_mean = float(np.mean(cur))
    base_mean = float(np.mean(base))
    
    # Calculate mean ratio
    if base_mean == 0:
        ratio_score = float("inf") if cur_mean != 0 else 1.0
    else:
        ratio_score = max(abs(cur_mean / base_mean), abs(base_mean / cur_mean)) if cur_mean != 0 else float("inf")
    
    # Calculate KS test if samples are large enough
    ks_stat = 0.0
    p_value = 1.0
    if cur.size >= 4 and base.size >= 4:
        ks_res = stats.ks_2samp(cur, base)
        ks_stat = float(ks_res.statistic)
        p_value = float(ks_res.pvalue)
    
    is_anomaly = bool(ratio_score >= ratio_threshold or (p_value < p_value_threshold and ks_stat >= 0.35))
    score = float(max(ratio_score if np.isfinite(ratio_score) else 100.0, ks_stat * 10.0))
    
    return {
        "is_anomaly": is_anomaly,
        "score": score,
        "method": "ks_and_ratio",
        "reason": f"mean_ratio={ratio_score:.2f}, ks_stat={ks_stat:.3f}, p_value={p_value:.4f}",
        "ks_stat": ks_stat,
        "p_value": p_value,
    }


