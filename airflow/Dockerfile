# OwnFeed orchestration image: Airflow + dbt (own virtualenv) + the ML dependencies.
FROM apache/airflow:3.3.2-python3.12

# dbt in its own virtualenv (Cosmos best practice: keeps dbt's dependencies away from Airflow's)
RUN python -m venv /home/airflow/dbt_venv \
 && /home/airflow/dbt_venv/bin/pip install --no-cache-dir dbt-core==1.12.6 dbt-duckdb==1.11.1 duckdb==2.0.0

# OwnFeed's Python dependencies. CPU-only PyTorch: Docker on a Mac can't use the Apple GPU.
# Same versions as the project's .venv so artifacts and the DuckDB file are interchangeable.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "apache-airflow==3.3.2" astronomer-cosmos duckdb==2.0.0 pandas==3.0.6 \
        numpy==2.5.4 sentence-transformers==6.2.0
