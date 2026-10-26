-- The user tower may use `history` only if it was collected BEFORE the log window, i.e. it never changes between
-- a user's page views within a source folder. Fails for any user whose history changes.
select source, user_id, count(distinct history) as versions
from {{ ref('stg_behaviors') }}
group by all
having count(distinct history) > 1
