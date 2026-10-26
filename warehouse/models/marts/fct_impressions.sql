-- One row per (page view, article shown), with the click label, the article's bucket and the time split.
select
    i.impression_key,
    i.source,
    i.user_id,
    i.time,
    i.date,
    i.hour,
    i.position,
    i.news_id,
    i.clicked,
    a.bucket,
    case
        when i.source = 'test' then 'test'
        when i.date < date '{{ var("validation_day") }}' then 'train'
        else 'validation'
    end as split
from {{ ref('int_impressions') }} as i
left join {{ ref('dim_articles') }} as a using (news_id)
