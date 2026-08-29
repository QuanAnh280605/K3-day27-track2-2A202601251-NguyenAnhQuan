from student_api import detect_metric


def test_large_volume_drop_is_anomaly():
    history = [1000, 1010, 995, 1008, 1004, 1012, 998]
    result = detect_metric(300, history, method="zscore")
    assert result["is_anomaly"] is True


def test_stable_value_is_not_anomaly():
    history = [1000, 1010, 995, 1008, 1004, 1012, 998]
    result = detect_metric(1002, history, method="zscore")
    assert result["is_anomaly"] is False


def test_mad_detector_handles_zero_mad():
    history = [100.0, 100.0, 100.0, 100.0, 100.0]
    # Identical value should not be anomaly
    assert detect_metric(100.0, history, method="mad")["is_anomaly"] is False
    # Deviant value should be caught
    assert detect_metric(150.0, history, method="mad")["is_anomaly"] is True


def test_context_aware_segment_history():
    overall_history = [1000, 1020, 1010, 990, 1005]  # weekday volume
    weekend_history = [250, 260, 245, 255, 250]      # saturday volume
    # Testing 250 with weekend segment context should be normal
    result = detect_metric(
        250,
        overall_history,
        method="auto",
        context={"same_segment_history": weekend_history, "day_of_week": 5},
    )
    assert result["is_anomaly"] is False

