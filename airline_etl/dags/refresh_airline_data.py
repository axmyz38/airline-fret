from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from airline_etl.pipeline import run_pipeline

with DAG(
    dag_id='refresh_airline_data',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['airline'],
) as dag:

    refresh = PythonOperator(
        task_id='run_pipeline',
        python_callable=run_pipeline,
    )
