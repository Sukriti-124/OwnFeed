"""OwnFeed retraining (manual trigger): train a challenger, compare it with the champion, promote only if it wins.

embed_new_articles -> train_challenger -> evaluate_and_promote

The model behind the reported results is pinned as "v1-paper" and never overwritten. Training and model selection
never use the locked test day (Nov 15): pipeline/ml.py refuses it.
"""
from datetime import datetime

from airflow.sdk import Param, dag, task


@dag(
    schedule=None,  # run from the UI ("Trigger DAG") or `airflow dags trigger ownfeed_retrain`
    start_date=datetime(2019, 11, 9),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    params={
        "train_until": Param("2019-11-14", type="string", description="Train on days before this date"),
        "eval_day": Param("2019-11-14", type="string", description="Day used to compare challenger vs champion (not the test day)"),
    },
    tags=["ownfeed", "ml"],
    doc_md=__doc__,
)
def ownfeed_retrain():
    @task
    def embed_new_articles():
        from pipeline.ml import embed_new_articles

        return embed_new_articles()

    @task
    def train_challenger(params=None):
        from pipeline.ml import train_challenger

        return train_challenger(params["train_until"])

    @task
    def evaluate_and_promote(challenger: str, params=None):
        from pipeline.ml import compare_and_promote

        return compare_and_promote(challenger, params["eval_day"])

    embedded = embed_new_articles()
    challenger = train_challenger()
    embedded >> challenger
    evaluate_and_promote(challenger)


ownfeed_retrain()
