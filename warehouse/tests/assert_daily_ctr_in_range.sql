-- Sanity check: each day's click-through rate should be between 2% and 6% (MIND is ~3.8–4.1%).
select date, avg(clicked) as ctr
from {{ ref('fct_impressions') }}
group by date
having avg(clicked) not between 0.02 and 0.06
