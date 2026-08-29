"""Robust, context-aware anomaly detectors used through the stable student API."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _safe_float(val: Any) -> float | None:
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
    if history is None:
        return []

    if isinstance(history, pd.DataFrame):
        df = history.copy()
        target_col = None
        if context and "metric_name" in context and context["metric_name"] in df.columns:
            target_col = context["metric_name"]
        else:
            for c in ["value", "metric", "row_count", "count", "amount", "total"]:
                if c in df.columns:
                    target_col = c
                    break
            if target_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                target_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

        dow_key = None
        for k in ["day_of_week", "dow", "weekday"]:
            if context and k in context and context[k] is not None and k in df.columns:
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

    if isinstance(history, dict) and not isinstance(history, pd.DataFrame):
        return _clean_history(list(history.values()), context=context)

    if isinstance(history, pd.Series):
        return [float(x) for x in history.dropna() if _safe_float(x) is not None]

    if isinstance(history, np.ndarray):
        flat = history.flatten()
        return [float(x) for x in flat if _safe_float(x) is not None]

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


def zscore_detector(current: Any, history: Iterable[Any], threshold: float = 3.0, *args: Any, **kwargs: Any) -> dict[str, Any]:
    curr_val = _safe_float(current)
    if curr_val is None:
        return {"is_anomaly": True, "score": float("inf"), "method": "zscore", "reason": "current_value_nan_or_infinite"}
    clean = _clean_history(history)
    if len(clean) < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "zscore", "reason": "insufficient_history"}
    values = np.asarray(clean, dtype=float)
    mean, std = float(np.mean(values)), float(np.std(values))
    score = float("inf") if std == 0 and curr_val != mean else (0.0 if std == 0 else abs(curr_val - mean) / std)
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "zscore",
        "reason": f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}",
    }


def mad_detector(current: Any, history: Iterable[Any], threshold: float = 3.5, *args: Any, **kwargs: Any) -> dict[str, Any]:
    curr_val = _safe_float(current)
    if curr_val is None:
        return {"is_anomaly": True, "score": float("inf"), "method": "mad", "reason": "current_value_nan_or_infinite"}
    clean = _clean_history(history)
    if len(clean) < 5:
        return {"is_anomaly": False, "score": 0.0, "method": "mad", "reason": "insufficient_history"}
    values = np.asarray(clean, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    score = float("inf") if mad == 0 and curr_val != median else (0.0 if mad == 0 else 0.6745 * abs(curr_val - median) / mad)
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "mad",
        "reason": f"median={median:.3f}, mad={mad:.3f}, threshold={threshold}",
    }


def iqr_detector(current: Any, history: Iterable[Any], threshold: float = 1.5, *args: Any, **kwargs: Any) -> dict[str, Any]:
    curr_val = _safe_float(current)
    if curr_val is None:
        return {"is_anomaly": True, "score": float("inf"), "method": "iqr", "reason": "current_value_nan_or_infinite"}
    clean = _clean_history(history)
    if len(clean) < 4:
        return {"is_anomaly": False, "score": 0.0, "method": "iqr", "reason": "insufficient_history"}
    values = np.asarray(clean, dtype=float)
    q25 = float(np.percentile(values, 25))
    q75 = float(np.percentile(values, 75))
    iqr = q75 - q25
    if iqr == 0:
        return mad_detector(curr_val, history, threshold=threshold)
    dist = (q25 - curr_val) if curr_val < q25 else ((curr_val - q75) if curr_val > q75 else 0.0)
    score = dist / iqr
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "iqr",
        "reason": f"q25={q25:.3f}, q75={q75:.3f}, iqr={iqr:.3f}, threshold={threshold}",
    }


def ewma_detector(current: Any, history: Iterable[Any], threshold: float = 3.0, alpha: float = 0.3, *args: Any, **kwargs: Any) -> dict[str, Any]:
    curr_val = _safe_float(current)
    if curr_val is None:
        return {"is_anomaly": True, "score": float("inf"), "method": "ewma", "reason": "current_value_nan_or_infinite"}
    clean = _clean_history(history)
    if len(clean) < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "ewma", "reason": "insufficient_history"}
    weights = np.array([(1.0 - alpha) ** i for i in range(len(clean))][::-1], dtype=float)
    weights /= np.sum(weights)
    values = np.asarray(clean, dtype=float)
    ewma_mean = float(np.sum(weights * values))
    ewma_var = float(np.sum(weights * ((values - ewma_mean) ** 2)))
    ewma_std = float(np.sqrt(max(0.0, ewma_var)))
    score = 0.0 if ewma_std == 0 and curr_val == ewma_mean else (float("inf") if ewma_std == 0 else abs(curr_val - ewma_mean) / ewma_std)
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "ewma",
        "reason": f"ewma_mean={ewma_mean:.3f}, ewma_std={ewma_std:.3f}, threshold={threshold}",
    }


def detect_anomaly(
    current: Any,
    history: Any,
    method: str | None = "auto",
    threshold: float = 3.0,
    context: dict[str, Any] | None = None,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    curr_val = _safe_float(current)
    if curr_val is None:
        return {"is_anomaly": True, "score": float("inf"), "method": str(method or "auto"), "reason": "current_value_nan_or_infinite"}

    context = context or {}

    # Extract threshold override from context if present
    effective_thresh = float(threshold)
    if "threshold" in context and context["threshold"] is not None:
        try:
            effective_thresh = float(context["threshold"])
        except (ValueError, TypeError):
            pass

    norm_method = str(method or "auto").lower().strip().replace("-", "_").replace(" ", "_")

    if norm_method in {"zscore", "z_score", "std", "normal", "z"}:
        return zscore_detector(curr_val, history, threshold=effective_thresh, **kwargs)
    if norm_method in {"mad", "median_absolute_deviation", "modified_z", "modified_zscore"}:
        mad_th = effective_thresh if ("threshold" in context or threshold != 3.0) else max(effective_thresh, 3.5)
        return mad_detector(curr_val, history, threshold=mad_th, **kwargs)
    if norm_method in {"iqr", "interquartile_range", "tukey", "boxplot"}:
        return iqr_detector(curr_val, history, threshold=1.5 if effective_thresh == 3.0 else effective_thresh, **kwargs)
    if norm_method in {"ewma", "exponential_smoothing", "ema"}:
        return ewma_detector(curr_val, history, threshold=effective_thresh, **kwargs)

    # 1. Check known event suppression
    known_event = context.get("known_event") or context.get("event")
    if known_event and str(known_event).lower() not in {"none", "false", "0", ""}:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "auto:known_event",
            "reason": f"suppressed_for_known_event={known_event}",
        }

    # 2. Check explicit bounds
    min_bound = _safe_float(context.get("min_value") or context.get("min"))
    max_bound = _safe_float(context.get("max_value") or context.get("max"))
    if min_bound is not None and curr_val < min_bound:
        return {"is_anomaly": True, "score": float("inf"), "method": "auto:bounds", "reason": f"below_min={min_bound}"}
    if max_bound is not None and curr_val > max_bound:
        return {"is_anomaly": True, "score": float("inf"), "method": "auto:bounds", "reason": f"above_max={max_bound}"}

    # 3. Check explicit segment history
    segment = context.get("same_segment_history") or context.get("segment_history")
    if segment is not None:
        segment_clean = _clean_history(segment)
        if len(segment_clean) >= 5:
            mad_th = effective_thresh if ("threshold" in context or threshold != 3.0) else max(effective_thresh, 3.5)
            result = mad_detector(curr_val, segment_clean, threshold=mad_th, **kwargs)
            result["method"] = "auto:seasonal_mad"
            result["reason"] += f"; segment_size={len(segment_clean)}"
            return result
        if len(segment_clean) >= 3:
            result = zscore_detector(curr_val, segment_clean, threshold=effective_thresh, **kwargs)
            result["method"] = "auto:seasonal_zscore"
            result["reason"] += f"; segment_size={len(segment_clean)}"
            return result

    # 4. Check day of week seasonality
    clean_hist = _clean_history(history, context=context)
    dow_raw = context.get("day_of_week") or context.get("dow") or context.get("weekday")
    if dow_raw is not None and len(clean_hist) >= 14:
        try:
            dow_val = int(dow_raw) % 7
            cand_direct = [clean_hist[i] for i in range(len(clean_hist)) if (i % 7) == dow_val]
            cand_lookback = [
                clean_hist[i]
                for i in range(len(clean_hist))
                if ((dow_val - (len(clean_hist) - i)) % 7) == dow_val
            ]
            mad_th = effective_thresh if ("threshold" in context or threshold != 3.0) else max(effective_thresh, 3.5)
            if len(cand_direct) >= 5:
                res = mad_detector(curr_val, cand_direct, threshold=mad_th, **kwargs)
                res["method"] = "auto:seasonal_mad"
                res["reason"] += f"; seasonal_dow={dow_val}"
                return res
            elif len(cand_direct) >= 3:
                res = zscore_detector(curr_val, cand_direct, threshold=effective_thresh, **kwargs)
                res["method"] = "auto:seasonal_zscore"
                res["reason"] += f"; seasonal_dow={dow_val}"
                return res
            elif len(cand_lookback) >= 5:
                res = mad_detector(curr_val, cand_lookback, threshold=mad_th, **kwargs)
                res["method"] = "auto:seasonal_mad"
                res["reason"] += f"; seasonal_dow={dow_val}"
                return res
            elif len(cand_lookback) >= 3:
                res = zscore_detector(curr_val, cand_lookback, threshold=effective_thresh, **kwargs)
                res["method"] = "auto:seasonal_zscore"
                res["reason"] += f"; seasonal_dow={dow_val}"
                return res
        except (ValueError, TypeError):
            pass

    # 5. Check linear trend if present
    eff_len = len(clean_hist)
    trend_ctx = context.get("trend")
    has_trend = bool(trend_ctx and str(trend_ctx).lower() not in {"none", "false", "0", ""})
    if eff_len >= 7:
        x_axis = np.arange(eff_len, dtype=float)
        y_axis = np.asarray(clean_hist, dtype=float)
        poly = np.polyfit(x_axis, y_axis, 1)
        slope, intercept = float(poly[0]), float(poly[1])
        preds = slope * x_axis + intercept
        residuals = y_axis - preds
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_axis - np.mean(y_axis)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

        if (r2 > 0.70 and abs(slope) > 0.01) or has_trend:
            expected_next = slope * eff_len + intercept
            res_mad = float(np.median(np.abs(residuals - np.median(residuals))))
            scale = res_mad if res_mad > 1e-9 else (float(np.std(residuals)) if float(np.std(residuals)) > 1e-9 else 1.0)
            trend_score = 0.6745 * abs(curr_val - expected_next) / scale
            is_trend_anomaly = bool(trend_score > (effective_thresh if "threshold" in context else max(effective_thresh, 3.5)))
            return {
                "is_anomaly": is_trend_anomaly,
                "score": float(trend_score),
                "method": "auto:trend",
                "reason": f"trend: expected={expected_next:.2f}, actual={curr_val:.2f}, score={trend_score:.2f}",
            }

    # 6. Default MAD or Z-score
    if eff_len >= 5:
        mad_th = effective_thresh if ("threshold" in context or threshold != 3.0) else max(effective_thresh, 3.5)
        result = mad_detector(curr_val, clean_hist, threshold=mad_th, **kwargs)
        result["method"] = "auto:mad"
    elif eff_len >= 3:
        result = zscore_detector(curr_val, clean_hist, threshold=effective_thresh, **kwargs)
        result["method"] = "auto:zscore"
    else:
        result = {
            "is_anomaly": False,
            "score": 0.0,
            "method": "auto",
            "reason": "insufficient_history",
        }

    # Apply directional filter if specified
    if context and result.get("is_anomaly", False) and "direction" in context:
        direction = str(context["direction"]).lower()
        baseline_val = float(np.median(clean_hist)) if len(clean_hist) > 0 else 0.0
        if direction in {"drop", "lower", "down"} and curr_val > baseline_val:
            result["is_anomaly"] = False
        elif direction in {"spike", "upper", "up"} and curr_val < baseline_val:
            result["is_anomaly"] = False

    return result
