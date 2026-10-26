-- One row per article across both source folders (an article can appear in both; keep the train copy).
with ranked as (
    select
        news_id,
        category,
        subcategory,
        title,
        coalesce(abstract, '') as abstract,
        source as first_seen_in,
        row_number() over (partition by news_id order by source = 'train' desc) as copy_rank
    from {{ source('bronze', 'news') }}
)
select news_id, category, subcategory, title, abstract, first_seen_in
from ranked
where copy_rank = 1
