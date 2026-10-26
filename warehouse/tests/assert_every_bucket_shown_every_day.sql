-- Every slider bucket must have articles shown every day, or the arranger couldn't fill its slots.
with days as (select distinct date from {{ ref('fct_impressions') }}),
     buckets as (select distinct bucket from {{ ref('bucket_mapping') }}),
     shown as (select date, bucket, count(distinct news_id) as articles from {{ ref('fct_impressions') }} group by all)
select d.date, b.bucket
from days as d cross join buckets as b
left join shown as s on s.date = d.date and s.bucket = b.bucket
where coalesce(s.articles, 0) = 0
