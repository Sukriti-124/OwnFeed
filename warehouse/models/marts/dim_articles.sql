-- Every article with its slider bucket: a subcategory override wins, otherwise the category's bucket.
-- Articles in unmapped categories (kids, middleeast, northamerica) get bucket = null.
with overrides as (
    select subcategory, bucket from {{ ref('bucket_mapping') }} where subcategory is not null
),
categories as (
    select category, bucket from {{ ref('bucket_mapping') }} where subcategory is null
)
select
    n.news_id,
    n.category,
    n.subcategory,
    coalesce(o.bucket, c.bucket) as bucket,
    n.title,
    n.abstract,
    n.first_seen_in
from {{ ref('stg_news') }} as n
left join overrides as o on o.subcategory = n.subcategory
left join categories as c on c.category = n.category
