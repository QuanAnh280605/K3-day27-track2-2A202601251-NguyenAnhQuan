"""Production-grade anomaly detection with context awareness, robust statistics, trend fitting, and seasonal segmentation.

Supports:
- Z-score baseline detector
- Robust Median Absolute Deviation (MAD) detector with zero-MAD edge handling
- Interquartile Range (IQR) detector (Tukey's fences)
- Exponentially Weighted Moving Average (EWMA) detector
- Rolling window detector
- Intelligent `auto` mode handling seasonality (day_of_week), segment histories, known events, trends, bounds, and directional filters
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _safe_float(val: Any) -> float | None:
    """Safely convert any input to a finite float, returning None if invalid."""
    if val is None:
        return None
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _clean_history(history: Any, context: dict[str, Any] | None = None) -> list[float]:
    """Extract clean finite float series from dicts, lists, arrays, Series, or DataFrames."""
    if history is None:
        return []

    # Handle pandas DataFrame
    if isinstance(history, pd.DataFrame):
        df = history.copy()
        target_col = None
        if context and "metric_name" in context and context["metric_name"] in df.columns:
            target_col = context["metric_name"]
        else:
            candidates = ["value", "metric", "row_count", "count", "amount", "total", "score"]
            for c in candidates:
                if c in df.columns:
                    target_col = c
                    break
            if target_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                target_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

        # Apply day_of_week filtering if present in DataFrame
        dow_key = None
        for k in ["day_of_week", "dow", "weekday"]:
            if context and k in context and context[k] is not None and k in df.columns:
                dow_key = k
                break
            elif k in df.columns and context and ("day_of_week" in context or "dow" in context or "weekday" in context):
                dow_key = k
                break

        if dow_key and context:
            dow_val = context.get("day_of_week") or context.get("dow") or context.get("weekday")
            if dow_val is not None:
                try:
                    df = df[df[dow_key] == int(dow_val)]
                except (ValueError, TypeError):
                    pass

        series = df[target_col]
        return [float(x) for x in series.dropna() if _safe_float(x) is not None]

    # Handle dictionary mapping (e.g. {date: value} or {index: value})
    if isinstance(history, dict) and not isinstance(history, pd.DataFrame):
        vals = list(history.values())
        return _clean_history(vals, context=context)

    # Handle pandas Series
    if isinstance(history, pd.Series):
        return [float(x) for x in history.dropna() if _safe_float(x) is not None]

    # Handle numpy ndarray
    if isinstance(history, np.ndarray):
        flat = history.flatten()
        return [float(x) for x in flat if _safe_float(x) is not None]

    # Handle lists, tuples, or general iterables
    out: list[float] = []
    try:
        for item in history:
            if isinstance(item, dict):
                val = None
                if context and "metric_name" in context and context["metric_name"] in item:
                    val = item[context["metric_name"]]
                else:
                    for k in ["value", "metric", "row_count", "count", "amount", "total"]:
                        if k in item:
                            val = item[k]
                            break
                    if val is None and len(item) > 0:
                        val = next(iter(item.values()))
                sf = _safe_float(val)
            else:
                sf = _safe_float(item)
            if sf is not None:
                out.append(sf)
    except TypeError:
        return []
    return out


def zscore_detector(
    current: Any,
    history: Iterable[Any],
    threshold: float = 3.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Z-score anomaly detector."""
    curr_val = _safe_float(current)
    if curr_val is None:
        return {
            "is_anomaly": True,
            "score": float("inf"),
            "method": "zscore",
            "reason": "current_value_nan_or_infinite",
        }

    clean = _clean_history(history)
    if len(clean) < 3:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "zscore",
            "reason": "insufficient_history",
        }

    values = np.asarray(clean, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))

    if std == 0.0 or std < 1e-9:
        score = 0.0 if abs(curr_val - mean) < 1e-9 else float("inf")
    else:
        score = abs(curr_val - mean) / std

    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "zscore",
        "reason": f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}",
    }


def mad_detector(
    current: Any,
    history: Iterable[Any],
    threshold: float = 3.5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Robust Median Absolute Deviation (MAD) detector with zero-MAD edge case handling."""
    curr_val = _safe_float(current)
    if curr_val is None:
        return {
            "is_anomaly": True,
            "score": float("inf"),
            "method": "mad",
            "reason": "current_value_nan_or_infinite",
        }

    clean = _clean_history(history)
    if len(clean) < 3:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "mad",
            "reason": "insufficient_history",
        }

    values = np.asarray(clean, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if mad == 0.0 or mad < 1e-9:
        if abs(curr_val - median) < 1e-9:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": "mad",
                "reason": f"median={median:.3f}, mad=0.0 (identical baseline)",
            }
        relative_diff = abs(curr_val - median) / (abs(median) if abs(median) > 1e-9 else 1.0)
        score = float(max(relative_diff * 10.0, 10.0))
        return {
            "is_anomaly": True,
            "score": float(score),
            "method": "mad",
            "reason": f"median={median:.3f}, mad=0.0, deviation={curr_val - median:.3f}",
        }

    modified_z = 0.6745 * abs(curr_val - median) / mad
    return {
        "is_anomaly": bool(modified_z > threshold),
        "score": float(modified_z),
        "method": "mad",
        "reason": f"median={median:.3f}, mad={mad:.3f}, threshold={threshold}",
    }


def iqr_detector(
    current: Any,
    history: Iterable[Any],
    threshold: float = 1.5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Interquartile Range (IQR) boxplot outlier detector (Tukey's method)."""
    curr_val = _safe_float(current)
    if curr_val is None:
        return {
            "is_anomaly": True,
            "score": float("inf"),
            "method": "iqr",
            "reason": "current_value_nan_or_infinite",
        }

    clean = _clean_history(history)
    if len(clean) < 4:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "iqr",
            "reason": "insufficient_history",
        }

    values = np.asarray(clean, dtype=float)
    q25 = float(np.percentile(values, 25))
    q75 = float(np.percentile(values, 75))
    iqr = q75 - q25

    if iqr == 0.0 or iqr < 1e-9:
        return mad_detector(current, history, threshold=threshold)

    dist = 0.0
    if curr_val < q25:
        dist = q25 - curr_val
    elif curr_val > q75:
        dist = curr_val - q75

    score = dist / iqr
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "iqr",
        "reason": f"q25={q25:.3f}, q75={q75:.3f}, iqr={iqr:.3f}, threshold={threshold}",
    }


def ewma_detector(
    current: Any,
    history: Iterable[Any],
    threshold: float = 3.0,
    alpha: float = 0.3,
    **kwargs: Any,
) -> dict[str, Any]:
    """Exponentially Weighted Moving Average (EWMA) detector."""
    curr_val = _safe_float(current)
    if curr_val is None:
        return {
            "is_anomaly": True,
            "score": float("inf"),
            "method": "ewma",
            "reason": "current_value_nan_or_infinite",
        }

    clean = _clean_history(history)
    if len(clean) < 3:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "ewma",
            "reason": "insufficient_history",
        }

    weights = np.array([(1.0 - alpha) ** i for i in range(len(clean))][::-1], dtype=float)
    weights /= np.sum(weights)
    values = np.asarray(clean, dtype=float)

    ewma_mean = float(np.sum(weights * values))
    ewma_var = float(np.sum(weights * ((values - ewma_mean) ** 2)))
    ewma_std = float(np.sqrt(max(0.0, ewma_var)))

    if ewma_std == 0.0 or ewma_std < 1e-9:
        score = 0.0 if abs(curr_val - ewma_mean) < 1e-9 else float("inf")
    else:
        score = abs(curr_val - ewma_mean) / ewma_std

    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "ewma",
        "reason": f"ewma_mean={ewma_mean:.3f}, ewma_std={ewma_std:.3f}, threshold={threshold}",
    }


def rolling_detector(
    current: Any,
    history: Iterable[Any],
    threshold: float = 3.0,
    window: int = 14,
    **kwargs: Any,
) -> dict[str, Any]:
    """Rolling window MAD detector focusing on recent history."""
    clean = _clean_history(history)
    recent = clean[-window:] if len(clean) > window else clean
    res = mad_detector(current, recent, threshold=threshold, **kwargs)
    res["method"] = "rolling_mad"
    return res


def detect_anomaly(
    current: Any,
    history: Any,
    method: str | None = "auto",
    threshold: float | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Context-aware anomaly detector supporting zscore, mad, iqr, ewma, rolling, and auto modes."""
    # Resolve threshold from context if not explicitly passed
    if threshold is None and context:
        for k in ["threshold", "z_threshold", "mad_threshold", "score_threshold"]:
            if k in context and context[k] is not None:
                try:
                    threshold = float(context[k])
                    break
                except (ValueError, TypeError):
                    pass

    # Normalize method string
    raw_method = str(method or "auto").lower().strip().replace("-", "_").replace(" ", "_")

    # Resolve default thresholds per method
    if threshold is None:
        if any(x in raw_method for x in ["mad", "modified_z"]):
            norm_thresh = 3.5
        elif any(x in raw_method for x in ["iqr", "tukey", "box"]):
            norm_thresh = 1.5
        else:
            norm_thresh = 3.0
    else:
        norm_thresh = float(threshold)

    # Route explicit non-auto methods
    if any(raw_method == x for x in ["mad", "median_absolute_deviation", "modified_zscore", "modified_z"]):
        return mad_detector(current, history, threshold=norm_thresh, **kwargs)
    if any(raw_method == x for x in ["zscore", "z_score", "std", "standard_deviation", "normal", "z"]):
        return zscore_detector(current, history, threshold=norm_thresh, **kwargs)
    if any(raw_method == x for x in ["iqr", "interquartile_range", "tukey", "boxplot"]):
        return iqr_detector(current, history, threshold=norm_thresh, **kwargs)
    if any(raw_method == x for x in ["ewma", "exponential_smoothing", "exponential_moving_average", "ema"]):
        return ewma_detector(current, history, threshold=norm_thresh, **kwargs)
    if any(raw_method == x for x in ["rolling", "rolling_mad", "rolling_zscore"]):
        return rolling_detector(current, history, threshold=norm_thresh, **kwargs)

    # Auto / Seasonal / Robust mode
    curr_val = _safe_float(current)
    if curr_val is None:
        return {
            "is_anomaly": True,
            "score": float("inf"),
            "method": "auto",
            "reason": "current_value_nan_or_infinite",
        }

    # 1. Check explicit bound constraints in context
    if context:
        min_bound = None
        max_bound = None
        for k in ["min_value", "min", "lower_bound", "min_range"]:
            if k in context and context[k] is not None:
                min_bound = _safe_float(context[k])
                break
        for k in ["max_value", "max", "upper_bound", "max_range"]:
            if k in context and context[k] is not None:
                max_bound = _safe_float(context[k])
                break

        if min_bound is not None and curr_val < min_bound:
            return {
                "is_anomaly": True,
                "score": float("inf"),
                "method": "auto:bounds",
                "reason": f"below_min_bound: value={curr_val:.3f} < min={min_bound:.3f}",
            }
        if max_bound is not None and curr_val > max_bound:
            return {
                "is_anomaly": True,
                "score": float("inf"),
                "method": "auto:bounds",
                "reason": f"above_max_bound: value={curr_val:.3f} > max={max_bound:.3f}",
            }

    clean_hist = _clean_history(history, context=context)
    effective_history = clean_hist
    context_notes: list[str] = []

    # 2. Check for explicit same_segment_history in context
    seg_hist = None
    if context:
        for k in ["same_segment_history", "segment_history", "segment"]:
            if k in context and context[k]:
                seg_hist = context[k]
                break

    if seg_hist:
        seg_clean = _clean_history(seg_hist)
        if len(seg_clean) >= 3:
            effective_history = seg_clean
            context_notes.append("used_same_segment_history")

    # 3. Seasonality handling via day_of_week
    elif context:
        dow_raw = None
        for k in ["day_of_week", "dow", "weekday"]:
            if k in context and context[k] is not None:
                dow_raw = context[k]
                break

        if dow_raw is not None:
            try:
                dow_val = int(dow_raw) % 7
                if len(clean_hist) >= 14:
                    cand_direct = [clean_hist[i] for i in range(len(clean_hist)) if (i % 7) == dow_val]
                    cand_lookback = [
                        clean_hist[i]
                        for i in range(len(clean_hist))
                        if ((dow_val - (len(clean_hist) - i)) % 7) == dow_val
                    ]

                    if len(cand_direct) >= 3:
                        effective_history = cand_direct
                        context_notes.append(f"seasonal_dow={dow_val}")
                    elif len(cand_lookback) >= 3:
                        effective_history = cand_lookback
                        context_notes.append(f"seasonal_dow={dow_val}")
                    else:
                        is_target_weekend = (dow_val >= 5)
                        same_type_points = [
                            clean_hist[i]
                            for i in range(len(clean_hist))
                            if ((i % 7) >= 5) == is_target_weekend
                        ]
                        if len(same_type_points) >= 3:
                            effective_history = same_type_points
                            context_notes.append(f"seasonal_type={'weekend' if is_target_weekend else 'weekday'}")
            except (ValueError, TypeError):
                pass

    # 4. Trend modeling & Awareness
    trend_ctx = context.get("trend") if context else None
    has_trend_key = bool(trend_ctx and str(trend_ctx).lower() not in {"none", "false", "0", ""})
    
    # Evaluate linear trend fit if >= 7 points
    eff_len = len(effective_history)
    trend_fit_detected = False
    if eff_len >= 7:
        x_axis = np.arange(eff_len, dtype=float)
        y_axis = np.asarray(effective_history, dtype=float)
        # Linear regression
        poly = np.polyfit(x_axis, y_axis, 1)
        slope, intercept = float(poly[0]), float(poly[1])
        preds = slope * x_axis + intercept
        residuals = y_axis - preds
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_axis - np.mean(y_axis)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

        if (r2 > 0.70 and abs(slope) > 0.01) or has_trend_key:
            trend_fit_detected = True
            expected_next = slope * eff_len + intercept
            res_mad = float(np.median(np.abs(residuals - np.median(residuals))))
            if res_mad < 1e-9:
                res_std = float(np.std(residuals))
                scale = res_std if res_std > 1e-9 else (abs(expected_next) * 0.05 if abs(expected_next) > 1e-9 else 1.0)
                trend_score = abs(curr_val - expected_next) / scale
            else:
                trend_score = 0.6745 * abs(curr_val - expected_next) / res_mad

            is_trend_anomaly = bool(trend_score > norm_thresh)
            if not is_trend_anomaly:
                return {
                    "is_anomaly": False,
                    "score": float(trend_score),
                    "method": "auto:trend",
                    "reason": f"on_trend: expected={expected_next:.2f}, slope={slope:.2f}, r2={r2:.2f}, score={trend_score:.2f}",
                }
            else:
                return {
                    "is_anomaly": True,
                    "score": float(trend_score),
                    "method": "auto:trend",
                    "reason": f"broken_trend: expected={expected_next:.2f}, actual={curr_val:.2f}, slope={slope:.2f}, r2={r2:.2f}, score={trend_score:.2f}",
                }

    # 5. Handle known events (promotions, planned outages, maintenance)
    known_event = context.get("known_event") if context else None
    if not known_event and context:
        for k in ["event", "planned_event", "holiday"]:
            if k in context and context[k]:
                known_event = context[k]
                break

    has_known_event = bool(known_event and str(known_event).lower() not in {"none", "false", "0", ""})

    # 6. Select base statistical detector
    if eff_len >= 5:
        base_res = mad_detector(curr_val, effective_history, threshold=norm_thresh, **kwargs)
        chosen_method = "auto:mad"
    elif eff_len >= 3:
        base_res = zscore_detector(curr_val, effective_history, threshold=norm_thresh, **kwargs)
        chosen_method = "auto:zscore"
    else:
        base_res = {
            "is_anomaly": False,
            "score": 0.0,
            "method": "auto",
            "reason": "insufficient_history",
        }
        chosen_method = "auto"

    # 7. Apply known_event suppression
    if has_known_event:
        event_name = str(known_event)
        context_notes.append(f"known_event={event_name}")
        base_res["is_anomaly"] = False
        base_res["method"] = "auto:known_event_suppressed"
        base_res["reason"] = f"known_event={event_name}; deviation_expected; baseline_score={base_res['score']:.2f}"
    else:
        base_res["method"] = chosen_method

    # 8. Apply directional filter if specified
    if context and base_res.get("is_anomaly", False):
        dir_key = context.get("direction") or context.get("mode")
        if dir_key:
            direction = str(dir_key).lower()
            baseline_val = float(np.median(effective_history)) if eff_len > 0 else 0.0
            if direction in {"drop", "lower", "down", "negative"} and curr_val > baseline_val:
                base_res["is_anomaly"] = False
                context_notes.append("ignored_spike_per_direction_filter")
            elif direction in {"spike", "upper", "up", "increase", "positive"} and curr_val < baseline_val:
                base_res["is_anomaly"] = False
                context_notes.append("ignored_drop_per_direction_filter")

    # 9. Append context annotations to reason
    if context_notes:
        base_res["reason"] += f" [{', '.join(context_notes)}]"
    elif context:
        dow = context.get("day_of_week") or context.get("dow")
        metric = context.get("metric_name") or context.get("metric")
        base_res["reason"] += f" [dow={dow}, metric={metric}]"

    return base_res
