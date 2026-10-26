-- One row per (source folder, article): the same article can appear in both folders.
-- file_row keeps each article's line in the raw file (saved embeddings follow that order).
select
    source,
    news_id,
    category,
    subcategory,
    title,
    coalesce(abstract, '') as abstract,
    _file_row as file_row
from {{ source('bronze', 'news') }}
