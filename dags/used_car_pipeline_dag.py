"""DAG Airflow untuk pipeline yang sama dengan run_pipeline.py.

Airflow tidak berjalan native di Windows. File ini dipakai bila pipeline dijalankan di
WSL, Docker, atau server Linux: salin ke folder dags/ milik Airflow dan sesuaikan PROJECT.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/bda-assignment2"
PY = f"cd {PROJECT}/src && python"

default_args = {"retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="used_car_lakehouse_pipeline",
    start_date=datetime(2026, 10, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["bda", "lakehouse"],
) as dag:
    ingest = BashOperator(task_id="ingest_raw", bash_command=f"{PY} s01_ingest_raw.py")
    curate = BashOperator(task_id="raw_to_curated", bash_command=f"{PY} s02_raw_to_curated.py")
    serve = BashOperator(task_id="curated_to_serving", bash_command=f"{PY} s03_curated_to_serving.py")
    prep = BashOperator(task_id="prepare_ml_dataset", bash_command=f"{PY} s04_prepare_ml_dataset.py")
    sk = BashOperator(task_id="train_sklearn", bash_command=f"{PY} s05_train_sklearn.py")
    sp = BashOperator(task_id="train_spark_mllib", bash_command=f"{PY} s06_train_spark_mllib.py")
    ev = BashOperator(task_id="evaluate", bash_command=f"{PY} s07_evaluate.py")
    graph = BashOperator(task_id="graph_analytics", bash_command=f"{PY} s08_graph_analytics.py")

    ingest >> curate >> [serve, prep, graph]
    prep >> [sk, sp] >> ev
