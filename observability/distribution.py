from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import stats


def _numeric_values(values: Iterable[float] | Any) -> np.ndarray:
    parsed: list[float] = []
    if values is None:
        return np.array([], dtype=float)
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    result = np.asarray(parsed, dtype=float)
    return result[np.isfinite(result)]


def detect_distribution_shift(
    current_values: Iterable[float] | Any,
    baseline_values: Iterable[float] | Any,
    ratio_threshold: float = 3.0,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect location and shape drift with a scale-normalized quantile distance and KS test."""
    cur, base = _numeric_values(current_values), _numeric_values(baseline_values)
    if cur.size == 0 or base.size == 0:
        return {"is_anomaly": False, "score": 0.0, "method": "quantile_shift", "reason": "empty_input"}

    quantiles = np.linspace(0.05, 0.95, 19)
    base_q, cur_q = np.quantile(base, quantiles), np.quantile(cur, quantiles)
    base_iqr = float(np.subtract(*np.quantile(base, [0.75, 0.25])))
    fallback_scale = max(abs(float(np.median(base))) * 0.05, float(np.std(base)), 1e-9)
    scale = base_iqr if base_iqr > 1e-9 else fallback_scale
    location_shape_score = float(np.mean(np.abs(cur_q - base_q)) / scale)
    cur_iqr = float(np.subtract(*np.quantile(cur, [0.75, 0.25])))
    spread_score = abs(float(np.log((cur_iqr + 1e-9) / (base_iqr + 1e-9))))
    score = float(max(location_shape_score, spread_score))

    ks_stat = 0.0
    p_value = 1.0
    if cur.size >= 4 and base.size >= 4:
        try:
            ks_res = stats.ks_2samp(cur, base)
            ks_stat = float(ks_res.statistic)
            p_value = float(ks_res.pvalue)
        except Exception:
            pass

    is_anomaly = bool(score >= ratio_threshold or (p_value < 0.05 and ks_stat >= 0.35))
    return {
        "is_anomaly": is_anomaly,
        "score": score,
        "method": "quantile_shift",
        "ks_stat": ks_stat,
        "p_value": p_value,
        "reason": (
            f"baseline_median={np.median(base):.3f}, current_median={np.median(cur):.3f}, "
            f"baseline_iqr={base_iqr:.3f}, current_iqr={cur_iqr:.3f}, score={score:.3f}, threshold={ratio_threshold}"
        ),
    }
