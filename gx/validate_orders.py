#!/usr/bin/env python3
"""Great Expectations Core 1.21 Suite, ValidationDefinition, and Checkpoint.

Packages business expectations into an Expectation Suite and runs them
through a Checkpoint to evaluate data quality before downstream processing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import great_expectations as gx
except ImportError as exc:
    raise SystemExit("great_expectations is not installed. Run: pip install -r requirements.txt") from exc


def build_and_run_checkpoint(df: pd.DataFrame) -> tuple[bool, Any]:
    context = gx.get_context(mode="ephemeral")

    # 1. Build Expectation Suite
    suite = gx.ExpectationSuite(name="orders_expectation_suite")
    expectations = [
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="order_id", notes="order_id must never be null (Critical)"
        ),
        gx.expectations.ExpectColumnValuesToBeUnique(
            column="order_id", notes="order_id must be unique primary key (Critical)"
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="customer_id", notes="customer_id required (Critical)"
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="amount", min_value=0, notes="revenue amount must be non-negative (Critical)"
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="currency", value_set=["USD", "VND"], notes="currency must be USD or VND (Critical)"
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="status",
            value_set=["pending", "completed", "refunded", "cancelled"],
            notes="status enum validation (Warning)",
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="created_at", notes="order created timestamp required"
        ),
    ]
    for exp in expectations:
        suite.add_expectation(exp)

    context.suites.add(suite)

    # 2. Configure Data Source & Asset
    data_source = context.data_sources.add_pandas("orders_pandas_source")
    asset = data_source.add_dataframe_asset(name="orders_df_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("orders_batch_definition")

    # 3. Validation Definition
    validation_definition = gx.ValidationDefinition(
        name="orders_validation_definition",
        data=batch_definition,
        suite=suite,
    )
    context.validation_definitions.add(validation_definition)

    # 4. Checkpoint
    checkpoint = gx.Checkpoint(
        name="orders_checkpoint",
        validation_definitions=[validation_definition],
    )
    context.checkpoints.add(checkpoint)

    # 5. Run Checkpoint
    result = checkpoint.run(batch_parameters={"dataframe": df})
    return bool(result.success), result


def main() -> None:
    orders_path = ROOT / "data" / "incoming" / "orders.csv"
    df = pd.read_csv(orders_path)
    print(f"Validating {len(df)} orders from {orders_path.name}...")

    success, result = build_and_run_checkpoint(df)
    
    print("\n=== GREAT EXPECTATIONS CHECKPOINT SUMMARY ===")
    print(f"Overall Checkpoint Status : {'PASS' if success else 'FAIL'}")
    
    # Severity Action Decision
    if success:
        print("Action                    : ALLOW -> Proceed with downstream dbt build.")
    else:
        print("Action                    : BLOCK / QUARANTINE -> Critical expectations violated!")
        sys.exit(1)


if __name__ == "__main__":
    main()

