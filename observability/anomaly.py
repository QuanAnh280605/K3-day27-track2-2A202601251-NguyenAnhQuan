"""Production-grade anomaly detection with context awareness, robust statistics, and seasonal segmentation.

Supports:
- Z-score baseline detector
- Robust Median Absolute Deviation (MAD) detector with zero-MAD edge handling
- Interquartile Range (IQR) detector
- Exponentially Weighted Moving Average (EWMA) detector
- Intelligent `auto` mode handling seasonality (day_of_week), segment histories, known events, and trends
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _safe_float(val: Any) -> float | None:
    """Safely convert any input to a finite float, returning None if invalid."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _clean_history(history: Any, context: dict[str, Any] | None = None) -> list[float]:
    """Extract clean finite float series from lists, arrays, Series, or DataFrames."""
    if history is None:
        return []

    # Handle pandas DataFrame
    if isinstance(history, pd.DataFrame):
        df = history.copy()
        target_col = None
        if context and "metric_name" in context and context["metric_name"] in df.columns:
            target_col = context["metric_name"]
        else:
            candidates = ["value", "metric", "row_count", "count", "amount", "total"]
            for c in candidates:
                if c in df.columns:
                    target_col = c
                    break
            if target_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                target_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

        # Apply day_of_week filtering if present in DataFrame
        if context and "day_of_week" in context and "day_of_week" in df.columns:
            dow = context["day_of_week"]
            df = df[df["day_of_week"] == dow]

        series = df[target_col]
        return [float(x) for x in series.dropna() if _safe_float(x) is not None]

    # Handle pandas Series
    if isinstance(history, pd.Series):
        return [float(x) for x in history.dropna() if _safe_float(x) is not None]

    # Handle lists, tuples, numpy arrays, or general iterables
    out: list[float] = []
    try:
        for item in history:
            if isinstance(item, dict):
                val = None
                if context and "metric_name" in context and context["metric_name"] in item:
                    val = item[context["metric_name"]]
                else:
                    for k in ["value", "metric", "row_count", "count", "amount"]:
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
    """Interquartile Range (IQR) boxplot outlier detector."""
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


def detect_anomaly(
    current: Any,
    history: Any,
    method: str = "auto",
    threshold: float | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Context-aware anomaly detector supporting zscore, mad, iqr, ewma, and auto modes."""
    # Normalize method string
    norm_method = str(method).lower().strip().replace("-", "_").replace(" ", "_")

    # Resolve default thresholds per method
    if threshold is None:
        if "mad" in norm_method:
            norm_thresh = 3.5
        elif "iqr" in norm_method:
            norm_thresh = 1.5
        else:
            norm_thresh = 3.0
    else:
        norm_thresh = float(threshold)

    # Route explicit non-auto methods
    if norm_method in {"mad", "median_absolute_deviation"}:
        return mad_detector(current, history, threshold=norm_thresh, **kwargs)
    if norm_method in {"zscore", "z_score", "std"}:
        return zscore_detector(current, history, threshold=norm_thresh, **kwargs)
    if norm_method in {"iqr", "interquartile_range"}:
        return iqr_detector(current, history, threshold=norm_thresh, **kwargs)
    if norm_method in {"ewma", "exponential_smoothing"}:
        return ewma_detector(current, history, threshold=norm_thresh, **kwargs)

    if norm_method in {"auto", "seasonal", "seasonal_mad"}:
        curr_val = _safe_float(current)
        if curr_val is None:
            return {
                "is_anomaly": True,
                "score": float("inf"),
                "method": "auto",
                "reason": "current_value_nan_or_infinite",
            }

        clean_hist = _clean_history(history, context=context)
        effective_history = clean_hist
        context_notes: list[str] = []

        # 1. Check for explicit same_segment_history in context
        if context and "same_segment_history" in context and context["same_segment_history"]:
            seg_clean = _clean_history(context["same_segment_history"])
            if len(seg_clean) >= 3:
                effective_history = seg_clean
                context_notes.append("used_same_segment_history")

        # 2. Seasonality handling via day_of_week
        elif context and "day_of_week" in context and context["day_of_week"] is not None:
            try:
                dow_val = int(context["day_of_week"]) % 7
                # If history represents a multi-week sequential daily series (>= 14 points)
                if len(clean_hist) >= 14:
                    cand_direct = [clean_hist[i] for i in range(len(clean_hist)) if (i % 7) == dow_val]
                    cand_lookback = [
                        clean_hist[i]
                        for i in range(len(clean_hist))
                        if ((dow_val - (len(clean_hist) - i)) % 7) == dow_val
                    ]

                    # Prefer the candidate with >= 3 samples
                    if len(cand_direct) >= 3:
                        effective_history = cand_direct
                        context_notes.append(f"seasonal_dow={dow_val}")
                    elif len(cand_lookback) >= 3:
                        effective_history = cand_lookback
                        context_notes.append(f"seasonal_dow={dow_val}")
                    else:
                        # Extract same day-type (weekend vs weekday)
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

        # 3. Handle known events (promotions, planned outages, maintenance)
        known_event = context.get("known_event") if context else None
        has_known_event = bool(known_event and str(known_event).lower() not in {"none", "false", "0", ""})

        # 4. Select base detector: robust MAD if >= 5 samples, else zscore
        eff_len = len(effective_history)
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

        # 5. Apply known_event suppression
        if has_known_event:
            event_name = str(known_event)
            context_notes.append(f"known_event={event_name}")
            # Anomaly alert suppressed because behavior is attributable to planned event
            base_res["is_anomaly"] = False
            base_res["method"] = f"auto:known_event_suppressed"
            base_res["reason"] = f"known_event={event_name}; deviation_expected; baseline_score={base_res['score']:.2f}"
        else:
            base_res["method"] = chosen_method

        # 6. Apply directional filter if specified (e.g. only drop or only spike)
        if context and "direction" in context and base_res.get("is_anomaly", False):
            direction = str(context["direction"]).lower()
            baseline_val = float(np.median(effective_history)) if eff_len > 0 else 0.0
            if direction in {"drop", "lower", "down"} and curr_val > baseline_val:
                base_res["is_anomaly"] = False
                context_notes.append("ignored_spike_per_direction_filter")
            elif direction in {"spike", "upper", "up", "increase"} and curr_val < baseline_val:
                base_res["is_anomaly"] = False
                context_notes.append("ignored_drop_per_direction_filter")

        # 7. Append context annotations to reason
        if context_notes:
            base_res["reason"] += f" [{', '.join(context_notes)}]"
        elif context:
            dow = context.get("day_of_week")
            metric = context.get("metric_name")
            base_res["reason"] += f" [dow={dow}, metric={metric}]"

        return base_res

    raise ValueError(f"Unsupported method: {method}")
