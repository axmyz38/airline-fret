# trading-

This repository hosts miscellaneous experiments. It now includes an example ETL
pipeline for airline data.

## Airline ETL

The `airline_etl` package demonstrates how to:

1. Import flights, aircraft capacities and bookings from CSV files.
2. Validate and normalise data (date formats and weight units).
3. Store the results in a normalised SQLite database with `flights`, `aircraft`
   and `bookings` tables.
4. Refresh the data via a scheduled job using an Airflow DAG
   (`airline_etl/dags/refresh_airline_data.py`).

Run the pipeline manually:

```bash
python -m airline_etl.pipeline
```

An Airflow deployment can be configured to pick up the DAG for scheduled runs.
