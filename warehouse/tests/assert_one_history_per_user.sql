-- dim_user_history must have exactly one row per (source, user).
select source, user_id, count(*) as rows
from {{ ref('dim_user_history') }}
group by all
having count(*) > 1
