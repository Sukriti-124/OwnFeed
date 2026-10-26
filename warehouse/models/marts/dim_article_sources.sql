-- Which articles each source folder lists, in the raw file's order (used by src/data.py load_news).
select source, news_id, file_row
from {{ ref('stg_news') }}
