# Incident Report — E-Commerce Revenue & Support KB Ingestion Failure

## Severity
**P1 — Critical (Data Corruption & Stale AI Agent Responses)**

## Summary
On 2026-08-29, the e-commerce data pipeline reported a `SUCCESS` status, but downstream business monitoring alerted on two critical anomalies:
1. Significant daily revenue distortion and duplication in `fct_daily_revenue` serving the CEO Dashboard.
2. AI Support Agent retrieving outdated refund policy guidelines due to stale KB document ingestion.

## Detection
- **Signal 1:** Data Contract Violation (`unique` constraint on `order_id` in `orders.csv` and `freshness` SLA violation on `kb_documents.jsonl`).
- **Signal 2:** Statistical Volume Drop Anomaly (MAD detector detected partial ingestion keeping only 25% of rows, score=5.53).
- **Signal 3:** dbt Unit Test failure on `fct_daily_revenue` exposing revenue inflation during multi-version customer dimension joins.
- **First observed time:** 2026-08-29 09:35 UTC.

## Root Cause
1. **Duplicate PK & Partial Ingestion in Orders Ingestion:** Upstream batch ingestion experienced duplicate primary keys (`order_id`) and incomplete data dumps without schema/type validation before pipeline execution.
2. **SCD Type 2 Customer Join Multiplying Rows:** `fct_daily_revenue.sql` performed a naive `LEFT JOIN` on `stg_customers` filtered only by `is_active = true`. Customers with multiple active/historical records duplicated the completed order rows, inflating reported revenue.
3. **Stale Knowledge Base Sync:** `kb_documents.jsonl` was published with a 3-hour lag exceeding the 60-minute freshness contract, resulting in RAG indexing stale policy documents.

## Evidence
1. **Contract Failure Evidence:** `src/contract_validator.py` detected duplicate `order_id` values and flagged KB `published_at` timestamp freshness (>60 min delay).
2. **dbt Test Evidence:** Native dbt unit test `multiple_active_customer_versions_do_not_inflate_revenue` failed under non-deduplicated join, proving row inflation.
3. **Statistical Anomaly Evidence:** `observability/anomaly.py` MAD detector detected abnormal volume drop from baseline ~600 rows to 150 rows.
4. **SLO Breach Evidence:** `observability/slo.py` multi-window burn rate exceeded 14.4x (2% budget consumed in 1 hour), triggering a P1 Page.

## Blast Radius
```text
orders.csv / raw_orders
  └─► stg_orders (amount_usd, order_id)
        └─► fct_daily_revenue (daily_revenue, completed_order_rows)
              └─► ceo_revenue_dashboard (Executive KPIs)

kb_documents.jsonl / kb_documents
  └─► kb_active_docs (content)
        └─► rag_index (vector embeddings)
              └─► support_agent (Customer Chatbot Responses)
```

## Mitigation
1. **Deterministic Gatekeeping:** Enforced Great Expectations checkpoint and Python contract validator at ingestion boundary. Any critical schema/PK failure immediately blocks execution and isolates data to quarantine.
2. **dbt Transformation Protection:** Refactored `fct_daily_revenue.sql` with deduplicated active customer dimensions (`SELECT DISTINCT customer_id FROM stg_customers WHERE is_active = true`) and added singular business tests.
3. **Freshness Monitoring:** Automated freshness alerts on KB ingestion with warning severity and fallback caching.

## Recovery
- Cleaned incoming orders data and regenerated seed files (`make dbt`).
- Re-synced knowledge base documents with current UTC timestamps.
- Executed end-to-end dbt build with all 19 tests passing.

## Verification
- [x] Contract healthy: `validate_orders` passes with 0 failures on healthy baseline.
- [x] dbt tests healthy: 19/19 models, seeds, data tests, and unit tests passed.
- [x] Anomaly detector healthy: MAD & auto context-aware detector correctly handles weekday/weekend segments.
- [x] SLO healthy: Error budget calculation and multi-window burn rate verified.
- [x] Downstream output verified: `fct_daily_revenue` produces exact daily totals without multiplication.

## Prevention / Action Items
| Action | Owner | Deadline | Why |
|---|---|---|---|
| Enforce GX Checkpoint in CI/CD pipeline | Data Engineering | 2026-09-02 | Prevent malformed/duplicate orders from reaching staging |
| Add dbt native unit tests for all mart joins | Analytics Engineering | 2026-09-03 | Catch dimension multiplication before production deployment |
| Configure Google SRE Multi-window Burn Rate PagerDuty alerts | SRE / Reliability Team | 2026-09-05 | Eliminate alert fatigue and page only on sustained fast burn |
| Implement automated KB vector drift monitoring | AI Platform Team | 2026-09-07 | Detect embedding norm shift and stale retrieval immediately |

