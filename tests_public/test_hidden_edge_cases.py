import numpy as np
import pandas as pd
import pytest

from student_api import (
    column_downstream,
    detect_distribution,
    detect_metric,
    downstream_assets,
    multiwindow_burn,
    rag_embedding_shift,
    rag_length_shift,
    slo_status,
    validate_orders,
)


def test_contract_missing_required_and_optional_columns():
    contract = {
        "columns": {
            "order_id": {"required": True, "type": "integer"},
            "optional_note": {"required": False, "type": "string"},
        }
    }
    # Missing required
    df_missing_req = pd.DataFrame({"optional_note": ["hello"]})
    issues = validate_orders(df_missing_req, contract)
    assert any(i["check"] == "required_column" and i["column"] == "order_id" and not i["passed"] for i in issues)

    # Missing optional only -> should pass
    df_valid = pd.DataFrame({"order_id": [1, 2]})
    issues_valid = validate_orders(df_valid, contract)
    assert all(i["passed"] for i in issues_valid)


def test_contract_all_types_validation():
    contract = {
        "columns": {
            "int_col": {"type": "bigint", "required": True},
            "num_col": {"type": "double", "required": True},
            "str_col": {"type": "varchar", "required": True, "min_length": 2, "max_length": 10},
            "date_col": {"type": "timestamp", "required": True},
            "bool_col": {"type": "boolean", "required": True},
        }
    }
    # Valid types
    df_valid = pd.DataFrame({
        "int_col": [1, 2, 3],
        "num_col": [10.5, 20.0, 30.1],
        "str_col": ["abc", "hello", "world"],
        "date_col": ["2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z", "2026-08-03T00:00:00Z"],
        "bool_col": [True, False, True],
    })
    issues = validate_orders(df_valid, contract)
    assert all(i["passed"] for i in issues)

    # Invalid types
    df_invalid = pd.DataFrame({
        "int_col": [1.5, 2.0, 3.0],  # float where int expected
        "num_col": ["not_a_number", 20.0, 30.1],
        "str_col": ["a", "toolongstringexceedingmax", "valid"],  # min_length & max_length fail
        "date_col": ["invalid_date", "2026-08-02T00:00:00Z", "2026-08-03T00:00:00Z"],
        "bool_col": ["not_a_bool", False, True],
    })
    issues_invalid = [i for i in validate_orders(df_invalid, contract) if not i["passed"]]
    assert any(i["column"] == "int_col" and i["check"] == "type" for i in issues_invalid)
    assert any(i["column"] == "num_col" and i["check"] == "type" for i in issues_invalid)
    assert any(i["column"] == "str_col" and i["check"] == "min_length" for i in issues_invalid)
    assert any(i["column"] == "str_col" and i["check"] == "max_length" for i in issues_invalid)
    assert any(i["column"] == "date_col" and i["check"] == "type" for i in issues_invalid)
    assert any(i["column"] == "bool_col" and i["check"] == "type" for i in issues_invalid)


def test_contract_pattern_regex():
    contract = {
        "columns": {
            "sku": {"type": "string", "pattern": r"^SKU-\d{4}$"},
        }
    }
    df = pd.DataFrame({"sku": ["SKU-1234", "INVALID-SKU", "SKU-9999"]})
    issues = [i for i in validate_orders(df, contract) if not i["passed"]]
    assert any(i["column"] == "sku" and i["check"] == "pattern" for i in issues)


def test_contract_empty_df():
    contract = {
        "columns": {
            "order_id": {"type": "integer", "required": True},
        }
    }
    df_empty = pd.DataFrame(columns=["order_id"])
    issues = validate_orders(df_empty, contract)
    assert all(i["passed"] for i in issues)


def test_detect_metric_with_nans_and_infinities():
    history = [100.0, np.nan, 102.0, 98.0, None, 101.0, 99.0]
    res = detect_metric(100.0, history, method="auto")
    assert res["is_anomaly"] is False

    res_anomaly = detect_metric(300.0, history, method="auto")
    assert res_anomaly["is_anomaly"] is True


def test_detect_metric_case_insensitive_method():
    history = [100.0, 102.0, 98.0, 101.0, 99.0]
    res_z = detect_metric(100.0, history, method="ZSCORE")
    assert res_z["is_anomaly"] is False

    res_mad = detect_metric(100.0, history, method="  MAD  ")
    assert res_mad["is_anomaly"] is False


def test_detect_metric_small_history_safety():
    history = [100.0, 105.0]
    res = detect_metric(100.0, history, method="auto")
    assert res["is_anomaly"] is False
    assert "insufficient_history" in res["reason"]


def test_detect_distribution_with_nans_and_empty():
    cur = [100.0, np.nan, 102.0, None, 98.0, 101.0]
    base = [100.0, 101.0, 99.0, 100.5, np.nan]
    res = detect_distribution(cur, base)
    assert res["is_anomaly"] is False

    # Empty inputs
    assert detect_distribution([], [10.0])["is_anomaly"] is False
    assert detect_distribution([10.0], [])["is_anomaly"] is False


def test_slo_status_percentage_target_and_floats():
    res = slo_status(target=99.5, bad_events=2.0, total_events=100.0)
    assert res["allowed_bad_rate"] == pytest.approx(0.005)
    assert res["actual_bad_rate"] == pytest.approx(0.02)
    assert res["burn_rate"] == pytest.approx(4.0)
    assert res["breached"] is True


def test_multiwindow_burn_all_branches():
    # 1. Sustained 1h critical burn (>= 14.4)
    res1 = multiwindow_burn(short_window_burn=15.0, long_window_burn=15.0)
    assert res1["page"] is True and res1["severity"] == "critical"

    # 2. Sustained 6h fast burn (>= 6.0)
    res2 = multiwindow_burn(short_window_burn=7.0, long_window_burn=6.5)
    assert res2["page"] is True and res2["severity"] == "critical"

    # 3. Transient spike (short >= 6.0, long < 6.0) -> No page
    res3 = multiwindow_burn(short_window_burn=20.0, long_window_burn=1.5)
    assert res3["page"] is False and res3["severity"] == "warning"

    # 4. Sustained slow burn (short >= 1.0, long >= 1.0) -> Ticket / Warn
    res4 = multiwindow_burn(short_window_burn=1.5, long_window_burn=1.2)
    assert res4["page"] is False and res4["severity"] == "warning"

    # 5. Healthy
    res5 = multiwindow_burn(short_window_burn=0.2, long_window_burn=0.1)
    assert res5["page"] is False and res5["severity"] == "info"


def test_lineage_with_raw_root_dictionary():
    full_json = {
        "dataset_lineage": {
            "a": ["b"],
            "b": ["c"],
            "c": ["d"],
        },
        "column_lineage": {
            "a.col1": ["b.col1"],
            "b.col1": ["c.col1"],
        },
    }
    assert downstream_assets(full_json, "a") == ["b", "c", "d"]
    assert column_downstream(full_json, "a.col1") == ["b.col1", "c.col1"]


def test_rag_metrics_robustness():
    # Length shift with mixed text inputs
    cur_texts = ["word1 word2 word3", "hello world", None, 12345]
    base_means = [2.0, 2.5, 2.2, 2.3, 2.4, 2.1, 2.3]
    res_len = rag_length_shift(cur_texts, base_means)
    assert isinstance(res_len["is_anomaly"], bool)

    # Embedding shift with NaNs
    cur_norms = [1.0, np.nan, 0.99, 1.01]
    base_norms = [1.0, 0.99, 1.01, 1.0, np.nan, 1.02]
    res_emb = rag_embedding_shift(cur_norms, base_norms)
    assert res_emb["is_anomaly"] is False
