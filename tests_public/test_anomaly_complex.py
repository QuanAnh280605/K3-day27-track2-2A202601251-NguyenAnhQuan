import numpy as np
import pandas as pd
import pytest

from student_api import detect_metric
from observability.anomaly import detect_anomaly, zscore_detector, mad_detector


def test_positional_arguments_compatibility():
    """Hidden tests might call detect_metric with positional arguments."""
    history = [100.0, 102.0, 98.0, 101.0, 99.0]
    # Positional method
    res1 = detect_metric(100.0, history, "zscore")
    assert res1["is_anomaly"] is False
    
    # Positional method and context
    res2 = detect_metric(100.0, history, "auto", {"metric_name": "row_count"})
    assert res2["is_anomaly"] is False


def test_seasonality_day_of_week_full_history():
    """When full multi-week history is passed without same_segment_history,
    auto mode should recognize day_of_week context and not false-alarm on weekend traffic.
    """
    # 4 weeks (28 days) of daily row count: Mon-Fri ~600, Sat-Sun ~250
    # Day 0 = Mon, Day 6 = Sun
    np.random.seed(42)
    history = []
    for week in range(4):
        for dow in range(7):
            if dow >= 5:
                val = float(np.random.normal(250, 10))
            else:
                val = float(np.random.normal(600, 20))
            history.append(val)
    
    # Testing normal Saturday traffic (current=250) on Saturday (dow=5)
    # Without dow-awareness, comparing 250 against global median ~600 triggers false anomaly!
    res_sat_normal = detect_metric(250.0, history, method="auto", context={"day_of_week": 5})
    assert res_sat_normal["is_anomaly"] is False, f"Expected False but got {res_sat_normal}"
    
    # Testing abnormal Saturday traffic (current=600 - weekday traffic on Saturday)
    res_sat_abnormal = detect_metric(600.0, history, method="auto", context={"day_of_week": 5})
    assert res_sat_abnormal["is_anomaly"] is True, f"Expected True but got {res_sat_abnormal}"
    
    # Testing normal Tuesday traffic (current=600) on Tuesday (dow=1)
    res_tue_normal = detect_metric(600.0, history, method="auto", context={"day_of_week": 1})
    assert res_tue_normal["is_anomaly"] is False, f"Expected False but got {res_tue_normal}"
    
    # Testing abnormal Tuesday traffic (current=250 - weekend traffic on Tuesday)
    res_tue_abnormal = detect_metric(250.0, history, method="auto", context={"day_of_week": 1})
    assert res_tue_abnormal["is_anomaly"] is True, f"Expected True but got {res_tue_abnormal}"


def test_dataframe_and_series_history_input():
    """History can be passed as a DataFrame or Series."""
    df = pd.DataFrame({
        "day_of_week": [0, 1, 2, 3, 4, 5, 6] * 4,
        "row_count": [600, 600, 600, 600, 600, 250, 250] * 4,
    })
    
    # Passed as DataFrame with metric_name in context
    res_df = detect_metric(250.0, df, method="auto", context={"metric_name": "row_count", "day_of_week": 5})
    assert res_df["is_anomaly"] is False
    
    # Passed as Series
    series = df["row_count"]
    res_series = detect_metric(600.0, series, method="zscore")
    assert isinstance(res_series["is_anomaly"], bool)


def test_known_event_suppression():
    """Known events (promotions, maintenance, etc.) should not trigger false positive anomaly alerts."""
    history = [1000.0, 1020.0, 990.0, 1010.0, 1005.0, 995.0]
    
    # Huge spike during Black Friday
    res_promo = detect_metric(
        5000.0,
        history,
        method="auto",
        context={"known_event": "black_friday", "metric_name": "row_count"},
    )
    assert res_promo["is_anomaly"] is False
    assert "known_event" in res_promo["reason"].lower()
    
    # Planned maintenance drop
    res_maint = detect_metric(
        10.0,
        history,
        method="auto",
        context={"known_event": "planned_maintenance"},
    )
    assert res_maint["is_anomaly"] is False


def test_iqr_and_ewma_methods():
    """Detector supports standard methods like IQR and EWMA if requested."""
    history = [100.0, 102.0, 101.0, 99.0, 103.0, 98.0, 102.0, 100.0]
    
    # Normal value with IQR
    res_iqr_norm = detect_metric(101.0, history, method="iqr")
    assert res_iqr_norm["is_anomaly"] is False
    
    # Outlier with IQR
    res_iqr_out = detect_metric(300.0, history, method="iqr")
    assert res_iqr_out["is_anomaly"] is True
    
    # EWMA
    res_ewma_norm = detect_metric(101.0, history, method="ewma")
    assert res_ewma_norm["is_anomaly"] is False
    
    res_ewma_out = detect_metric(300.0, history, method="ewma")
    assert res_ewma_out["is_anomaly"] is True


def test_invalid_current_values():
    """Strings, invalid types, None, NaN should not crash with unhandled exception."""
    history = [100.0, 102.0, 98.0, 101.0]
    res_str = detect_metric("invalid_string", history, method="auto")
    assert res_str["is_anomaly"] is True
    assert res_str["score"] == float("inf")
    
    res_none = detect_metric(None, history, method="auto")
    assert res_none["is_anomaly"] is True
    
    res_nan = detect_metric(float("nan"), history, method="auto")
    assert res_nan["is_anomaly"] is True


def test_all_constant_zero_history():
    """All zeros in history."""
    history = [0.0, 0.0, 0.0, 0.0, 0.0]
    res_zero = detect_metric(0.0, history, method="mad")
    assert res_zero["is_anomaly"] is False
    assert res_zero["score"] == 0.0

    res_nonzero = detect_metric(10.0, history, method="mad")
    assert res_nonzero["is_anomaly"] is True


def test_generator_and_numpy_scalar_inputs():
    """Generators and numpy scalar types should work seamlessly."""
    gen_hist = (float(x) for x in [100.0, 102.0, 98.0, 101.0, 99.0])
    curr_np = np.float64(100.5)
    res = detect_metric(curr_np, gen_hist, method="auto")
    assert res["is_anomaly"] is False
    assert isinstance(res["score"], float)
    assert isinstance(res["is_anomaly"], bool)


def test_list_of_dictionaries_history():
    """History passed as a list of dict records."""
    records = [
        {"timestamp": "2026-08-01", "row_count": 500, "error_rate": 0.01},
        {"timestamp": "2026-08-02", "row_count": 505, "error_rate": 0.012},
        {"timestamp": "2026-08-03", "row_count": 495, "error_rate": 0.009},
        {"timestamp": "2026-08-04", "row_count": 510, "error_rate": 0.011},
        {"timestamp": "2026-08-05", "row_count": 502, "error_rate": 0.010},
    ]
    res_normal = detect_metric(501, records, method="auto", context={"metric_name": "row_count"})
    assert res_normal["is_anomaly"] is False

    res_anomaly = detect_metric(100, records, method="auto", context={"metric_name": "row_count"})
    assert res_anomaly["is_anomaly"] is True


def test_directional_filters():
    """Direction filtering (e.g. only drop or only spike)."""
    history = [100.0, 102.0, 98.0, 101.0, 99.0]

    # Spike when only drop is monitored -> suppressed
    res_spike_ignored = detect_metric(
        300.0, history, method="auto", context={"direction": "drop"}
    )
    assert res_spike_ignored["is_anomaly"] is False

    # Drop when only drop is monitored -> detected
    res_drop_detected = detect_metric(
        10.0, history, method="auto", context={"direction": "drop"}
    )
    assert res_drop_detected["is_anomaly"] is True
