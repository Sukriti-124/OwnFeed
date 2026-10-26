-- Each user's past clicks (collected before the log window, so fixed per source folder) and the last N used
-- by the user tower.
with histories as (
    select distinct source, user_id, history
    from {{ ref('stg_behaviors') }}
),
lists as (
    select source, user_id,
           case when history = '' then []::varchar[] else string_split(history, ' ') end as all_clicks
    from histories
)
select
    source,
    user_id,
    len(all_clicks) as history_length,
    list_slice(all_clicks, greatest(len(all_clicks) - {{ var('history_len') }} + 1, 1), len(all_clicks)) as recent_clicks
from lists
