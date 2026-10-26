-- Every article once, with its slider bucket: a subcategory override wins, otherwise the category's bucket.
-- Articles in unmapped categories (kids, middleeast, northamerica) get bucket = null.
with articles as (
    select *, row_number() over (partition by news_id order by source = 'train' desc) as copy_rank
    from {{ ref('stg_news') }}
),
overrides as (
    select subcategory, bucket from {{ ref('bucket_mapping') }} where subcategory is not null
),
categories as (
    select category, bucket from {{ ref('bucket_mapping') }} where subcategory is null
)
select
    a.news_id,
    a.category,
    a.subcategory,
    coalesce(o.bucket, c.bucket) as bucket,
    a.title,
    a.abstract,
    a.source as first_seen_in
from articles as a
left join overrides as o on o.subcategory = a.subcategory
left join categories as c on c.category = a.category
where a.copy_rank = 1
