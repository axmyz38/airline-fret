# trading-

Trading

## Flight planning API

The repository now includes a simple flight planning module. Run the server:

```bash
python app.py
```

Send a POST request to `/plan` with JSON payload containing `dates`, `routes`
and `aircraft` lists to generate a schedule. Results are stored in the
`flight_schedule` SQLite table.
