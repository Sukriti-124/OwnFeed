-- stg_news must have each article at most once per source folder.
select source, news_id, count(*) as rows
from {{ ref('stg_news') }}
group by all
having count(*) > 1
