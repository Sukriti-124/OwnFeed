-- "Trending" for every article at every hour: clicks in the 24 hours BEFORE that hour (the current hour is
-- excluded, so nothing peeks ahead). Same rule as src/finder.py trending_scores; feeds log(1 + x) in the finder.
with hourly_clicks as (
    select hour, news_id, count(*) as clicks
    from {{ ref('fct_impressions') }}
    where clicked = 1
    group by all
),
article_span as (
    -- each article only needs hours from its first click to 24 h after its last click
    select news_id, min(hour) as first_hour, max(hour) + interval 24 hour as last_hour
    from hourly_clicks
    group by news_id
),
grid as (
    select s.news_id, unnest(generate_series(s.first_hour, s.last_hour, interval 1 hour)) as hour
    from article_span as s
),
filled as (
    select g.news_id, g.hour, coalesce(h.clicks, 0) as clicks_this_hour
    from grid as g
    left join hourly_clicks as h using (news_id, hour)
),
windowed as (
    select
        hour,
        news_id,
        clicks_this_hour,
        coalesce(sum(clicks_this_hour) over (
            partition by news_id order by hour rows between 24 preceding and 1 preceding
        ), 0) as clicks_last_24h
    from filled
)
select hour, news_id, clicks_this_hour, clicks_last_24h, ln(1 + clicks_last_24h) as trending_score
from windowed
where clicks_last_24h > 0 or clicks_this_hour > 0
