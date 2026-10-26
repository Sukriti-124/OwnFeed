-- One row per page view. impression_id restarts at 1 in each source folder, so the key is source + id.
with typed as (
    select
        case when _source_file like '%MINDsmall_test%' then 'test' else 'train' end as source,
        impression_id,
        user_id,
        time,
        history,
        impressions
    from {{ source('bronze', 'behaviors') }}
)
select
    source || '-' || impression_id as impression_key,
    source,
    impression_id,
    user_id,
    time,
    cast(time as date) as date,
    date_trunc('hour', time) as hour,
    history,
    impressions
from typed
