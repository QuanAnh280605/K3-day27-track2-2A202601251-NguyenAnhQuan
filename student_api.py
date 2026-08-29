"""Stable interface used by public and instructor-side hidden evaluation.

Students may refactor internals, but keep these function names and return shapes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from observability.anomaly import detect_anomaly
from observability.distribution import detect_distribution_shift
from observability.lineage import get_column_downstream, get_downstream_assets
from observability.rag_metrics import detect_embedding_norm_shift, detect_text_length_shift
from observability.slo import calculate_slo, evaluate_multiwindow_burn
from src.contract_validator import load_contract, validate_dataframe


def validate_orders(
    df: pd.DataFrame,
    contract_path: str | Path | dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    if isinstance(contract_path, dict):
        contract = contract_path
    else:
        contract = load_contract(contract_path)
    return validate_dataframe(df, contract, *args, **kwargs)


def detect_metric(
    current: float | Any,
    history: Iterable[float] | Any,
    method: str | None = "auto",
    context: dict[str, Any] | None = None,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    return detect_anomaly(current, history, method=method, context=context, *args, **kwargs)


def detect_distribution(
    current_values: Iterable[float] | Any,
    baseline_values: Iterable[float] | Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    return detect_distribution_shift(current_values, baseline_values, *args, **kwargs)


def slo_status(
    target: float | Any,
    bad_events: int | float,
    total_events: int | float,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    return calculate_slo(target, bad_events, total_events, *args, **kwargs)


def multiwindow_burn(
    short_window_burn: float = 0.0,
    long_window_burn: float = 0.0,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    return evaluate_multiwindow_burn(
        short_window_burn,
        long_window_burn,
        *args,
        **kwargs,
    )


def downstream_assets(
    graph: dict[str, list[str]] | Any,
    start: str,
    *args: Any,
    **kwargs: Any,
) -> list[str]:
    return get_downstream_assets(graph, start, *args, **kwargs)


def column_downstream(
    graph: dict[str, list[str]] | Any,
    start: str,
    *args: Any,
    **kwargs: Any,
) -> list[str]:
    return get_column_downstream(graph, start, *args, **kwargs)


def rag_length_shift(
    current_texts: Iterable[str] | Any,
    baseline_batch_means: Iterable[float] | Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    return detect_text_length_shift(current_texts, baseline_batch_means, *args, **kwargs)


def rag_embedding_shift(
    current_norms: Iterable[float] | Any,
    baseline_norms: Iterable[float] | Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    return detect_embedding_norm_shift(current_norms, baseline_norms, *args, **kwargs)
