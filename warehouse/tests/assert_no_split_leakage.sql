-- Fails if any row sits in the wrong time split (e.g. a validation or test day leaking into train).
select split, date, count(*) as rows
from {{ ref('fct_impressions') }}
where (split = 'train' and date >= date '{{ var("validation_day") }}')
   or (split = 'validation' and date <> date '{{ var("validation_day") }}')
   or (split = 'test' and date < date '{{ var("test_day") }}')
group by all
