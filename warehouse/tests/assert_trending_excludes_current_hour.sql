-- Leakage guard for the finder: trending at hour h must not count clicks from hour h itself.
-- Recompute an article's 24 h window independently for a sample of rows and compare.
with sample as (
    select * from {{ ref('fct_hourly_trending') }} using sample 2000 rows (reservoir, 42)
),
recomputed as (
    select s.hour, s.news_id, s.clicks_last_24h,
           (select count(*) from {{ ref('fct_impressions') }} as f
            where f.news_id = s.news_id and f.clicked = 1
              and f.hour >= s.hour - interval 24 hour and f.hour < s.hour) as expected
    from sample as s
)
select * from recomputed where clicks_last_24h <> expected
