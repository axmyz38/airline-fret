 codex/model-milp-problem-for-flight-scheduling
# trading-

codex/define-cost-function-and-expected-revenue
Trading tools and pricing utilities.

## Pricing API

Run the Flask application to expose the `/price` endpoint which returns an optimal
price based on remaining capacity. All price decisions are stored in a SQLite
`pricing_history` table.

```bash
python pricing_api.py
```

Send a request:

```bash
curl -X POST http://localhost:5000/price -H "Content-Type: application/json" \
     -d '{"remaining_capacity": 42}'
```
=======
Trading

## Flight planning API

The repository now includes a simple flight planning module. Run the server:

```bash
python app.py
```

Send a POST request to `/plan` with JSON payload containing `dates`, `routes`
and `aircraft` lists to generate a schedule. Results are stored in the
`flight_schedule` SQLite table.
=======
 main
 main
