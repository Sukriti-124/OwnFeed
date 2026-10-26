-- One row per page view, in the raw form the ML code expects (src/data.py load_behaviors).
select impression_key, source, impression_id, user_id, time, date, history, impressions
from {{ ref('stg_behaviors') }}
