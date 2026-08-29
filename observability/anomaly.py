"""Anomaly detection starter.

Z-score is deliberately the default baseline. Students should improve `auto`
mode for seasonality/outliers rather than deleting the simple implementation.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def zscore_detector(current: float, history: Iterable[float], threshold: float = 3.0) -> dict[str, Any]:
    if current is None or not np.isfinite(float(current)):
        return {"is_anomaly": True, "score": float("inf"), "method": "zscore", "reason": "current_value_nan_or_infinite"}
    
    clean = [float(x) for x in history if x is not None and np.isfinite(float(x))]
    values = np.asarray(clean, dtype=float)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "zscore", "reason": "insufficient_history"}
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        score = float("inf") if float(current) != mean else 0.0
    else:
        score = abs(float(current) - mean) / std
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "zscore",
        "reason": f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}",
    }


def mad_detector(current: float, history: Iterable[float], threshold: float = 3.5) -> dict[str, Any]:
    """Robust Median Absolute Deviation (MAD) detector with zero-MAD edge case handling."""
    if current is None or not np.isfinite(float(current)):
        return {"is_anomaly": True, "score": float("inf"), "method": "mad", "reason": "current_value_nan_or_infinite"}

    clean = [float(x) for x in history if x is not None and np.isfinite(float(x))]
    values = np.asarray(clean, dtype=float)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "mad", "reason": "insufficient_history"}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    
    if mad == 0:
        if float(current) == median:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": "mad",
                "reason": f"median={median:.3f}, mad=0.0 (identical baseline)",
            }
        # Non-zero difference from zero-variance baseline
        relative_diff = abs(float(current) - median) / (abs(median) if median != 0 else 1.0)
        score = relative_diff * 10.0  # Scale relative difference
        return {
            "is_anomaly": bool(score > threshold or current != median),
            "score": float(score),
            "method": "mad",
            "reason": f"median={median:.3f}, mad=0.0, deviation={float(current) - median:.3f}",
        }
    
    modified_z = 0.6745 * abs(float(current) - median) / mad
    return {
        "is_anomaly": bool(modified_z > threshold),
        "score": float(modified_z),
        "method": "mad",
        "reason": f"median={median:.3f}, mad={mad:.3f}, threshold={threshold}",
    }


def detect_anomaly(
    current: float,
    history: Iterable[float],
    *,
    method: str = "auto",
    threshold: float = 3.0,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Context-aware anomaly detector supporting zscore, mad, and auto modes."""
    norm_method = str(method).lower().strip()
    norm_thresh = float(threshold)

    if norm_method == "mad":
        return mad_detector(current, history, threshold=norm_thresh)
    if norm_method == "zscore":
        return zscore_detector(current, history, threshold=norm_thresh)
    
    if norm_method == "auto":
        clean_hist = [float(x) for x in history if x is not None and np.isfinite(float(x))]
        effective_history = clean_hist
        context_notes = []

        # 1. Apply context-provided segment history if available
        if context:
            if "same_segment_history" in context and context["same_segment_history"]:
                seg_clean = [float(x) for x in context["same_segment_history"] if x is not None and np.isfinite(float(x))]
                if len(seg_clean) >= 3:
                    effective_history = seg_clean
                    context_notes.append("used_same_segment_history")
            
            if "known_event" in context and context["known_event"]:
                event_name = context["known_event"]
                context_notes.append(f"known_event={event_name}")

        values = np.asarray(effective_history, dtype=float)
        
        # 2. Choose robust MAD if sufficient history (>=5 points), else fallback to zscore
        if values.size >= 5:
            result = mad_detector(current, effective_history, threshold=norm_thresh)
            result["method"] = "auto:mad"
        else:
            result = zscore_detector(current, effective_history, threshold=norm_thresh)
            result["method"] = "auto:zscore"
        
        if context_notes:
            result["reason"] += f" [{', '.join(context_notes)}]"
        elif context:
            dow = context.get("day_of_week")
            metric = context.get("metric_name")
            result["reason"] += f" [dow={dow}, metric={metric}]"
            
        return result

    raise ValueError(f"Unsupported method: {method}")


