-- Singular test: completed_order_rows must be strictly positive in revenue mart
select *
from {{ ref('fct_daily_revenue') }}
where completed_order_rows <= 0
