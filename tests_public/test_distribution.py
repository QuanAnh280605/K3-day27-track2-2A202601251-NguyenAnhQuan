from student_api import detect_distribution


def test_extreme_mean_shift_detected():
    baseline = [9, 10, 11, 10, 10]
    current = [190, 200, 210, 205]
    assert detect_distribution(current, baseline)["is_anomaly"] is True


def test_identical_distribution_not_detected():
    baseline = [10.0, 11.0, 10.5, 9.8, 10.2, 10.1]
    current = [10.1, 10.2, 9.9, 10.4, 10.0, 10.3]
    assert detect_distribution(current, baseline)["is_anomaly"] is False

