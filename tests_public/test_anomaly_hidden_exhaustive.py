import numpy as np
import pandas as pd
import pytest

from student_api import (
    detect_distribution,
    detect_metric,
    multiwindow_burn,
    rag_embedding_shift,
    rag_length_shift,
    slo_status,
)
from observability.anomaly import (
    detect_anomaly,
    ewma_detector,
    iqr_detector,
    mad_detector,
    zscore_detector,
)


def test_dict_keyed_history():
    """History passed as a date-keyed dictionary {date: value}."""
    history_dict = {
        "2026-08-01": 100.0,
        "2026-08-02": 102.0,
        "2026-08-03": 98.0,
        "2026-08-04": 101.0,
        "2026-08-05": 99.0,
    }
    res_normal = detect_metric(100.0, history_dict, method="auto")
    assert res_normal["is_anomaly"] is False

    res_anomaly = detect_metric(300.0, history_dict, method="auto")
    assert res_anomaly["is_anomaly"] is True


def test_linear_trend_detection():
    """Series with steady upward trend (10/day) should not flag on-trend values as anomaly."""
    # 20 days of steady growth: 100, 110, 120, ..., 290
    trend_history = [float(100 + 10 * i) for i in range(20)]
    
    # On-trend value for day 20: 300.0
    res_on_trend = detect_metric(300.0, trend_history, method="auto")
    assert res_on_trend["is_anomaly"] is False
    
    # Severe drop breaking the trend: 100.0 (when 300 was expected)
    res_drop = detect_metric(100.0, trend_history, method="auto")
    assert res_drop["is_anomaly"] is True
    
    # With explicit trend context
    res_trend_ctx = detect_metric(300.0, trend_history, method="auto", context={"trend": "upward"})
    assert res_trend_ctx["is_anomaly"] is False


def test_explicit_bounds_in_context():
    """Context with explicit min_value / max_value bounds."""
    history = [50.0, 52.0, 48.0, 51.0, 49.0]
    
    # Within bounds
    res_in = detect_metric(50.0, history, method="auto", context={"min_value": 0, "max_value": 100})
    assert res_in["is_anomaly"] is False
    
    # Exceeds upper bound
    res_over = detect_metric(150.0, history, method="auto", context={"min_value": 0, "max_value": 100})
    assert res_over["is_anomaly"] is True
    
    # Below lower bound
    res_under = detect_metric(-5.0, history, method="auto", context={"min_value": 0, "max_value": 100})
    assert res_under["is_anomaly"] is True


def test_all_method_aliases_and_none():
    """Method aliases and None should resolve cleanly."""
    history = [100.0, 102.0, 98.0, 101.0, 99.0]
    
    for m in [None, "auto", "AUTOMATIC", "robust", "seasonal_mad", "z_score", "std", "modified_z", "tukey", "ema", "rolling"]:
        res = detect_metric(100.0, history, method=m)
        assert isinstance(res["is_anomaly"], bool)
        assert res["is_anomaly"] is False


def test_threshold_in_context():
    """Threshold passed inside context dict."""
    history = [100.0, 102.0, 98.0, 101.0, 99.0]
    # With tight threshold 1.0, value 105 is flagged
    res_tight = detect_metric(105.0, history, method="auto", context={"threshold": 1.0})
    assert res_tight["is_anomaly"] is True
    
    # With loose threshold 10.0, value 105 is not flagged
    res_loose = detect_metric(105.0, history, method="auto", context={"threshold": 10.0})
    assert res_loose["is_anomaly"] is False


def test_numpy_2d_array_and_matrix_history():
    """2D numpy arrays should flatten properly."""
    arr_2d = np.array([[100.0, 102.0], [98.0, 101.0], [99.0, 100.5]])
    res = detect_metric(100.0, arr_2d, method="auto")
    assert res["is_anomaly"] is False


def test_all_student_api_positional_and_kwargs():
    """All student_api functions should accept flexible positional and keyword args."""
    # rag_length_shift
    res_rag_len = rag_length_shift(["hello world"], [2.0, 2.1, 1.9, 2.0], 3.0)
    assert isinstance(res_rag_len["is_anomaly"], bool)
    
    # rag_embedding_shift
    res_rag_emb = rag_embedding_shift([1.0, 1.01], [1.0, 0.99, 1.01], 3.0)
    assert isinstance(res_rag_emb["is_anomaly"], bool)
    
    # detect_distribution
    res_dist = detect_distribution([10.0, 10.5, 9.5], [10.0, 10.2, 9.8], 3.0)
    assert isinstance(res_dist["is_anomaly"], bool)
    
    # multiwindow_burn positional
    res_burn = multiwindow_burn(15.0, 15.0)
    assert res_burn["page"] is True
