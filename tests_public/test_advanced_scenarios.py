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
from src.contract_validator import determine_action, failed_issues


def test_anomaly_outlier_contamination_mad_vs_zscore():
    """Prove MAD is robust against outlier-contaminated history where Z-score fails."""
    # Baseline history with 1 single huge outlier (e.g. data glitch)
    polluted_history = [100.0, 102.0, 101.0, 99.0, 103.0, 1000.0, 100.0, 101.0]
    
    # Value 160 is clearly an anomaly compared to standard ~100 values
    current_value = 160.0
    
    # MAD ignores the 1000.0 outlier and catches the 160.0 anomaly
    mad_result = detect_metric(current_value, polluted_history, method="mad")
    assert mad_result["is_anomaly"] is True
    
    # Auto mode selects robust MAD for history >= 5
    auto_result = detect_metric(current_value, polluted_history, method="auto")
    assert auto_result["is_anomaly"] is True
    assert "mad" in auto_result["method"]


def test_distribution_bimodal_to_unimodal_shift():
    """Detect shape drift where mean remains identical but distribution structure changes."""
    # Baseline is bimodal: half around 10, half around 90 (mean = 50.0)
    baseline = [10.0, 11.0, 9.0, 10.5, 90.0, 91.0, 89.0, 90.5] * 5
    # Current is tightly unimodal around 50.0 (mean = 50.0)
    current = [49.0, 50.0, 51.0, 50.5, 49.5, 50.2, 49.8, 50.1] * 5
    
    # Means are virtually identical, but KS-test catches the major structural shift
    result = detect_distribution(current, baseline)
    assert result["is_anomaly"] is True
    assert result["ks_stat"] > 0.35


def test_distribution_variance_explosion():
    """Detect variance expansion where mean is preserved."""
    np.random.seed(42)
    baseline = np.random.normal(loc=100.0, scale=1.0, size=50).tolist()
    current = np.random.normal(loc=100.0, scale=25.0, size=50).tolist()
    
    result = detect_distribution(current, baseline)
    assert result["is_anomaly"] is True


def test_lineage_diamond_dependency_graph():
    """Diamond graph A -> B -> D and A -> C -> D returns unique BFS order."""
    diamond_graph = {
        "orders_raw": ["stg_orders", "stg_audit_orders"],
        "stg_orders": ["fct_revenue"],
        "stg_audit_orders": ["fct_revenue"],
        "fct_revenue": ["ceo_dashboard"],
    }
    downstream = downstream_assets(diamond_graph, "orders_raw")
    # All downstream nodes present without duplicates
    assert downstream == ["stg_orders", "stg_audit_orders", "fct_revenue", "ceo_dashboard"]


def test_lineage_cyclic_graph_termination():
    """Cyclic dependencies terminate safely without infinite recursion."""
    cyclic_graph = {
        "node_a": ["node_b"],
        "node_b": ["node_c"],
        "node_c": ["node_a", "node_d"],
    }
    downstream = downstream_assets(cyclic_graph, "node_a")
    assert set(downstream) == {"node_b", "node_c", "node_d"}


def test_lineage_deep_chain_traversal():
    """10-hop deep dependency chain traversed completely."""
    deep_graph = {f"L{i}": [f"L{i+1}"] for i in range(10)}
    downstream = downstream_assets(deep_graph, "L0")
    expected = [f"L{i}" for i in range(1, 11)]
    assert downstream == expected


def test_contract_large_scale_vectorization():
    """High performance validation on 10,000 rows."""
    n_rows = 10000
    df_large = pd.DataFrame({
        "order_id": np.arange(1, n_rows + 1),
        "amount": np.random.uniform(10.0, 500.0, size=n_rows),
        "currency": np.random.choice(["USD", "VND"], size=n_rows),
        "status": np.random.choice(["pending", "completed"], size=n_rows),
        "created_at": ["2026-08-28T10:00:00Z"] * n_rows,
        "updated_at": ["2026-08-28T10:05:00Z"] * n_rows,
        "customer_id": [f"C{i:05d}" for i in range(n_rows)],
    })
    contract = {
        "columns": {
            "order_id": {"type": "integer", "required": True, "unique": True, "severity": "critical"},
            "amount": {"type": "number", "min": 0, "required": True, "severity": "critical"},
            "currency": {"type": "string", "accepted_values": ["USD", "VND"], "severity": "critical"},
            "customer_id": {"type": "string", "required": True, "min_length": 3, "severity": "critical"},
        }
    }
    issues = validate_orders(df_large, contract)
    assert all(i["passed"] for i in issues)


def test_contract_timezone_variations():
    """Various ISO 8601 timezone offsets parsed correctly without error."""
    contract = {"columns": {"ts": {"type": "datetime", "required": True}}}
    df = pd.DataFrame({
        "ts": [
            "2026-08-28T10:00:00Z",
            "2026-08-28T17:00:00+07:00",
            "2026-08-28T05:00:00-05:00",
        ]
    })
    issues = validate_orders(df, contract)
    assert all(i["passed"] for i in issues)


def test_contract_min_max_boundary_conditions():
    """Exact min/max boundary values pass."""
    contract = {
        "columns": {
            "val": {"type": "number", "min": 10.0, "max": 100.0}
        }
    }
    df = pd.DataFrame({"val": [10.0, 50.0, 100.0]})
    issues = validate_orders(df, contract)
    assert all(i["passed"] for i in issues)

    # Exceeding boundary by small epsilon
    df_fail = pd.DataFrame({"val": [9.999, 100.001]})
    issues_fail = [i for i in validate_orders(df_fail, contract) if not i["passed"]]
    assert len(issues_fail) == 1


def test_slo_boundary_exact_burn_rates():
    """Boundary conditions for Google SRE burn rate thresholds."""
    # Exact 14.4x 1h threshold
    res_14_4 = multiwindow_burn(short_window_burn=14.4, long_window_burn=14.4)
    assert res_14_4["page"] is True

    # Exact 6.0x 6h threshold
    res_6_0 = multiwindow_burn(short_window_burn=6.0, long_window_burn=6.0)
    assert res_6_0["page"] is True

    # Exact 1.0x 3d threshold
    res_1_0 = multiwindow_burn(short_window_burn=1.0, long_window_burn=1.0)
    assert res_1_0["page"] is False
    assert res_1_0["severity"] == "warning"


def test_slo_spike_recovery_behavior():
    """Spike recovery: short window has cleared, long window still elevated -> do not page."""
    res = multiwindow_burn(short_window_burn=0.5, long_window_burn=14.0)
    assert res["page"] is False
    assert res["severity"] == "warning"
    assert "recovering" in res["reason"]


def test_rag_unicode_and_multilingual_tokens():
    """Support Vietnamese text with diacritics and emojis in token length approximation."""
    vietnamese_texts = [
        "Đơn hàng của tôi đã được giao thành công 🎉",
        "Chính sách đổi trả trong vòng 30 ngày làm việc",
        "Xin chào, tôi cần hỗ trợ kiểm tra trạng thái thanh toán",
    ]
    base_means = [10.0, 11.0, 10.5, 10.8, 10.2, 11.2, 10.7]
    result = rag_length_shift(vietnamese_texts, base_means)
    assert result["is_anomaly"] is False
    assert result["current_mean"] > 0



def test_rag_l2_norm_collapse():
    """Detect when vector embeddings collapse to near zero norms."""
    baseline_unit_norms = [1.0, 0.999, 1.001, 1.0, 0.998, 1.002]
    collapsed_norms = [0.001, 0.002, 0.0, 0.001]
    result = rag_embedding_shift(collapsed_norms, baseline_unit_norms)
    assert result["is_anomaly"] is True
    assert result["score"] > 10.0


def test_contract_mixed_severity_and_action():
    """Test mixed severities and operational action routing."""
    issues = [
        {"check": "not_null", "column": "order_id", "severity": "critical", "passed": False},
        {"check": "status", "column": "status", "severity": "warning", "passed": False},
        {"check": "info_check", "column": "notes", "severity": "info", "passed": False},
    ]
    critical_only = failed_issues(issues, min_severity="critical")
    assert len(critical_only) == 1
    assert critical_only[0]["column"] == "order_id"

    warning_and_above = failed_issues(issues, min_severity="warning")
    assert len(warning_and_above) == 2

    # Action determination
    assert determine_action(issues) == "block"
    assert determine_action(warning_and_above[1:]) == "quarantine"
    assert determine_action([]) == "pass"
