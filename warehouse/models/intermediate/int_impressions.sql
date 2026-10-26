-- Unpack each page view's "N123-1 N456-0 ..." list into one row per article shown.
with lists as (
    select impression_key, source, user_id, time, date, hour, string_split(impressions, ' ') as items
    from {{ ref('stg_behaviors') }}
),
unpacked as (
    select impression_key, source, user_id, time, date, hour,
           unnest(items) as item,
           generate_subscripts(items, 1) as position
    from lists
)
select
    impression_key, source, user_id, time, date, hour, position,
    split_part(item, '-', 1) as news_id,
    cast(split_part(item, '-', 2) as integer) as clicked
from unpacked
