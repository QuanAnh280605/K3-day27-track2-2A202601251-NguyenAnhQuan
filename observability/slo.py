from __future__ import annotations

from typing import Any


def calculate_slo(target: float, bad_events: int | float, total_events: int | float) -> dict[str, Any]:
    target = float(target)
    if 1.0 < target <= 100.0:
        target = target / 100.0

    if not 0 < target < 1:
        raise ValueError("target must be between 0 and 1 (exclusive)")
    
    bad = int(bad_events)
    total = int(total_events)
    if bad < 0 or total < 0 or bad > total:
        raise ValueError("invalid event counts")
    allowed_bad_rate = 1.0 - target
    if total == 0:

        return {
            "target": target,
            "actual_bad_rate": 0.0,
            "allowed_bad_rate": allowed_bad_rate,
            "burn_rate": 0.0,
            "remaining_error_budget_fraction": 1.0,
            "breached": False,
        }
    actual_bad_rate = bad_events / total_events
    burn_rate = actual_bad_rate / allowed_bad_rate
    consumed_fraction = min(1.0, actual_bad_rate / allowed_bad_rate)
    return {
        "target": target,
        "actual_bad_rate": actual_bad_rate,
        "allowed_bad_rate": allowed_bad_rate,
        "burn_rate": burn_rate,
        "remaining_error_budget_fraction": max(0.0, 1.0 - consumed_fraction),
        "breached": bool(actual_bad_rate > allowed_bad_rate),
    }


def evaluate_multiwindow_burn(
    short_window_burn: float = 0.0,
    long_window_burn: float = 0.0,
    policy: str = "sre",
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate multi-window multi-burn-rate alerting policy based on Google SRE principles.
    
    Prevents alert fatigue from transient spikes by requiring both short and long
    window burn rates to exceed thresholds before paging.
    
    Standard SRE thresholds for a 30-day budget:
    - 2% budget in 1 hour: short >= 14.4 and long >= 14.4 -> Page (Critical)
    - 5% budget in 6 hours: short >= 6.0 and long >= 6.0 -> Page (Critical)
    - 10% budget in 3 days: short >= 1.0 and long >= 1.0 -> Ticket / Warn (Warning)
    - Transient spike (short >= 6.0, long < 6.0) -> No Page (Warning)
    """
    short = float(short_window_burn)
    long = float(long_window_burn)

    # 1. Sustained critical 1-hour fast burn (consumes 2% budget in 1h)
    if short >= 14.4 and long >= 14.4:
        return {
            "page": True,
            "severity": "critical",
            "reason": "sustained_critical_burn_1h (>=14.4x across both windows)",
            "short_window_burn": short,
            "long_window_burn": long,
        }

    # 2. Sustained fast 6-hour burn (consumes 5% budget in 6h)
    if short >= 6.0 and long >= 6.0:
        return {
            "page": True,
            "severity": "critical",
            "reason": "sustained_fast_burn_6h (>=6.0x across both windows)",
            "short_window_burn": short,
            "long_window_burn": long,
        }

    # 3. Transient spike: short window high, but long window has not accumulated sufficient burn
    if short >= 6.0 and long < 6.0:
        return {
            "page": False,
            "severity": "warning",
            "reason": "transient_spike_no_page (short burn high but long burn below threshold)",
            "short_window_burn": short,
            "long_window_burn": long,
        }

    # 4. Spike recovery: long window still has past burn, but short window has dropped/recovered
    if short < 6.0 and long >= 6.0:
        return {
            "page": False,
            "severity": "warning",
            "reason": "spike_recovering_no_page (short window has cleared below threshold)",
            "short_window_burn": short,
            "long_window_burn": long,
        }

    # 5. Sustained slow burn (10% in 3 days -> ticket, no page)
    if short >= 1.0 and long >= 1.0:

        return {
            "page": False,
            "severity": "warning",
            "reason": "sustained_slow_burn_ticket (>=1.0x budget consumption)",
            "short_window_burn": short,
            "long_window_burn": long,
        }

    # 5. Normal / within budget
    return {
        "page": False,
        "severity": "info",
        "reason": "burn_rate_healthy",
        "short_window_burn": short,
        "long_window_burn": long,
    }

