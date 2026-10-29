"""OwnFeed daily pipeline: one run per day of MIND data (Nov 9–15, 2019, replayed with catch-up).

ingest_day -> dbt (each model + its tests) -> parity_check -> embed_new_articles -> evaluate_day -> refresh_demo_assets

A failed data test or parity check stops the run before anything downstream (model scores, demo) is touched.
"""
import os
from datetime import datetime

from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowFailException
from cosmos import DbtTaskGroup, ExecutionConfig, ProfileConfig, ProjectConfig, RenderConfig
from cosmos.constants import TestBehavior

WAREHOUSE_DIR = "/opt/ownfeed/warehouse"


@dag(
    schedule="@daily",
    start_date=datetime(2019, 11, 9),
    end_date=datetime(2019, 11, 15),
    catchup=True,
    max_active_runs=1,
    max_active_tasks=1,  # DuckDB allows one writer at a time
    tags=["ownfeed", "data", "ml"],
    doc_md=__doc__,
)
def ownfeed_daily():
    @task
    def ingest_day(ds=None):
        """Raw MIND TSV -> bronze Parquet for this run's day (idempotent: reloading replaces the day)."""
        from pipeline.ingest import ingest

        ingest([ds])

    dbt_warehouse = DbtTaskGroup(
        group_id="dbt_warehouse",
        project_config=ProjectConfig(WAREHOUSE_DIR),
        profile_config=ProfileConfig(profile_name="ownfeed", target_name="local",
                                     profiles_yml_filepath=f"{WAREHOUSE_DIR}/profiles.yml"),
        execution_config=ExecutionConfig(dbt_executable_path=os.environ.get("OWNFEED_DBT_EXECUTABLE", "dbt")),
        render_config=RenderConfig(test_behavior=TestBehavior.AFTER_EACH, should_detach_multiple_parents_tests=True),
        operator_args={"install_deps": False},
    )

    @task
    def parity_check():
        """Warehouse vs. the raw-file Python code: must match exactly (pipeline/check_parity.py)."""
        from pipeline import check_parity

        try:
            check_parity.main()
        except SystemExit as exit_code:
            if exit_code.code:
                raise AirflowFailException("Parity check failed: the warehouse no longer matches the raw-file code")

    @task
    def embed_new_articles():
        """MiniLM vectors for articles that don't have one yet."""
        from pipeline.ml import embed_new_articles

        return embed_new_articles()

    @task
    def evaluate_day(ds=None):
        """Score the champion model (and trending) on this day -> results/model_monitoring.csv."""
        from pipeline.ml import evaluate_day

        return evaluate_day(ds)

    @task
    def refresh_demo_assets():
        """Rebuild the demo's precomputed files from the latest warehouse."""
        from src.feed import build_demo_assets

        build_demo_assets()

    ingest_day() >> dbt_warehouse >> parity_check() >> embed_new_articles() >> evaluate_day() >> refresh_demo_assets()


ownfeed_daily()
