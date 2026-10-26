"""Phase 1 · Ingestion: raw MIND TSV files -> typed Parquet in the bronze layer of the lake.

    python -m pipeline.ingest --all                # every day and both news snapshots
    python -m pipeline.ingest --day 2019-11-14     # load or reload one day of behaviors

Bronze keeps the data faithful to the source (the impressions list stays one text column; unpacking it is a
transformation for the warehouse layer). Every row gets `_source_file` and `_ingested_at`. Loads are idempotent:
a partition is written to a temporary folder and swapped in, so reloading a day replaces it instead of duplicating it.
Each load is checked and recorded in lake/_load_log.csv.
"""
import argparse
import csv
import hashlib
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
LANDING = ROOT / "data"  # raw files exactly as downloaded
LAKE = ROOT / "lake"
BRONZE = LAKE / "bronze"
LOAD_LOG = LAKE / "_load_log.csv"

SOURCES = {"train": LANDING / "MINDsmall_train", "test": LANDING / "MINDsmall_test"}

NEWS_COLUMNS = {
    "news_id": "VARCHAR", "category": "VARCHAR", "subcategory": "VARCHAR", "title": "VARCHAR",
    "abstract": "VARCHAR", "url": "VARCHAR", "title_entities": "VARCHAR", "abstract_entities": "VARCHAR",
}
BEHAVIOR_COLUMNS = {"impression_id": "BIGINT", "user_id": "VARCHAR", "time": "VARCHAR", "history": "VARCHAR", "impressions": "VARCHAR"}
IMPRESSIONS_PATTERN = r"^N\d+-[01]( N\d+-[01])*$"  # e.g. "N123-1 N456-0"


class LoadError(Exception):
    """A check failed; nothing was written for this partition."""


def read_tsv(path: Path, columns: dict) -> str:
    """DuckDB expression that reads a headerless MIND TSV with explicit column types (no quoting in MIND files)."""
    cols = ", ".join(f"'{name}': '{kind}'" for name, kind in columns.items())
    return f"read_csv('{path}', delim='\\t', header=false, quote='', escape='', columns={{{cols}}})"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_load(table: str, partition: str, rows: int, source: Path, checksum: str, seconds: float, status: str, message: str = "") -> None:
    LAKE.mkdir(parents=True, exist_ok=True)
    new_file = not LOAD_LOG.exists()
    with open(LOAD_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["loaded_at", "table", "partition", "rows", "source_file", "source_md5", "seconds", "status", "message"])
        writer.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), table, partition, rows,
                         source.relative_to(ROOT), checksum, round(seconds, 2), status, message])


def write_partition(con: duckdb.DuckDBPyConnection, query: str, target_dir: Path) -> int:
    """Write `query` to target_dir/part-0.parquet via a temporary folder, then swap it in (idempotent)."""
    tmp_dir = target_dir.with_name(target_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    con.execute(f"COPY ({query}) TO '{tmp_dir / 'part-0.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    rows = con.execute(f"SELECT count(*) FROM '{tmp_dir / 'part-0.parquet'}'").fetchone()[0]
    shutil.rmtree(target_dir, ignore_errors=True)
    tmp_dir.rename(target_dir)
    return rows


def ingest_news(con: duckdb.DuckDBPyConnection, source: str) -> None:
    """One snapshot of the article catalog per source folder (MIND news files have no dates)."""
    path = SOURCES[source] / "news.tsv"
    start, checksum = time.time(), md5(path)
    try:
        con.execute(f"CREATE OR REPLACE TEMP VIEW raw_news AS SELECT * FROM {read_tsv(path, NEWS_COLUMNS)}")
        total, missing_ids, duplicate_ids = con.execute(
            "SELECT count(*), count(*) FILTER (WHERE news_id IS NULL), count(*) - count(DISTINCT news_id) FROM raw_news"
        ).fetchone()
        if total == 0 or missing_ids or duplicate_ids:
            raise LoadError(f"rows={total}, missing ids={missing_ids}, duplicate ids={duplicate_ids}")
        query = (f"SELECT *, '{path.relative_to(ROOT)}' AS _source_file, current_timestamp AS _ingested_at FROM raw_news")
        rows = write_partition(con, query, BRONZE / "news" / f"source={source}")
        log_load("news", f"source={source}", rows, path, checksum, time.time() - start, "success")
        print(f"✓ news  source={source:5s}  {rows:>7,} rows")
    except (LoadError, duckdb.Error) as e:
        log_load("news", f"source={source}", 0, path, checksum, time.time() - start, "failed", str(e))
        raise


def behaviors_view(con: duckdb.DuckDBPyConnection, source: str) -> Path:
    """Typed view over one source's behaviors file, with the timestamp parsed and a `date` column."""
    path = SOURCES[source] / "behaviors.tsv"
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW raw_behaviors_{source} AS
        SELECT impression_id, user_id,
               try_strptime(time, '%m/%d/%Y %I:%M:%S %p') AS time,
               coalesce(history, '') AS history,
               impressions,
               CAST(try_strptime(time, '%m/%d/%Y %I:%M:%S %p') AS DATE) AS date
        FROM {read_tsv(path, BEHAVIOR_COLUMNS)}
    """)
    return path


def ingest_behaviors_day(con: duckdb.DuckDBPyConnection, source: str, day: str, path: Path, checksum: str) -> None:
    """One day of page views -> bronze/behaviors/date=YYYY-MM-DD/."""
    start = time.time()
    view = f"raw_behaviors_{source}"
    try:
        total, bad_time, missing_keys, bad_impressions = con.execute(f"""
            SELECT count(*),
                   count(*) FILTER (WHERE time IS NULL),
                   count(*) FILTER (WHERE impression_id IS NULL OR user_id IS NULL),
                   count(*) FILTER (WHERE NOT regexp_full_match(impressions, '{IMPRESSIONS_PATTERN}'))
            FROM {view} WHERE date = DATE '{day}'
        """).fetchone()
        if total == 0 or bad_time or missing_keys or bad_impressions:
            raise LoadError(f"rows={total}, unparsable times={bad_time}, missing keys={missing_keys}, malformed impressions={bad_impressions}")
        query = (f"SELECT impression_id, user_id, time, history, impressions, '{path.relative_to(ROOT)}' AS _source_file, "
                 f"current_timestamp AS _ingested_at FROM {view} WHERE date = DATE '{day}'")
        rows = write_partition(con, query, BRONZE / "behaviors" / f"date={day}")
        log_load("behaviors", f"date={day}", rows, path, checksum, time.time() - start, "success")
        print(f"✓ behaviors date={day}  {rows:>7,} rows")
    except (LoadError, duckdb.Error) as e:
        log_load("behaviors", f"date={day}", 0, path, checksum, time.time() - start, "failed", str(e))
        raise


def ingest(days: list[str] | None = None) -> None:
    """Load the given days (None = every day in both sources, plus both news snapshots)."""
    missing = [str(p) for d in SOURCES.values() for p in (d / "news.tsv", d / "behaviors.tsv") if not p.exists()]
    if missing:
        sys.exit("Raw MIND files not found:\n  " + "\n  ".join(missing))

    con = duckdb.connect()
    if days is None:
        for source in SOURCES:
            ingest_news(con, source)

    for source in SOURCES:
        path = behaviors_view(con, source)
        checksum = md5(path)
        unparsable = con.execute(f"SELECT count(*) FROM raw_behaviors_{source} WHERE date IS NULL").fetchone()[0]
        if unparsable:
            log_load("behaviors", f"source={source}", 0, path, checksum, 0, "failed", f"{unparsable} rows with unparsable time")
            raise LoadError(f"{source}: {unparsable} rows with an unparsable time")
        available = [str(d) for (d,) in con.execute(f"SELECT DISTINCT date FROM raw_behaviors_{source} ORDER BY date").fetchall()]
        for day in available if days is None else [d for d in days if d in available]:
            ingest_behaviors_day(con, source, day, path, checksum)

    if days is not None:
        loaded = {p.name.removeprefix("date=") for p in (BRONZE / "behaviors").glob("date=*")}
        if missing_days := [d for d in days if d not in loaded]:
            sys.exit(f"No data for: {', '.join(missing_days)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="load every day and both news snapshots")
    group.add_argument("--day", action="append", metavar="YYYY-MM-DD", help="load one day (repeatable)")
    args = parser.parse_args()
    try:
        ingest(None if args.all else args.day)
    except LoadError as e:
        sys.exit(f"✗ Load failed: {e}  (see {LOAD_LOG.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
