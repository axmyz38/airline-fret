# trading-

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
