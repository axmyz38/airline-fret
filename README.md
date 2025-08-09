 codex/model-milp-problem-for-flight-scheduling
# trading-

codex/model-3d-cargo-hold-and-generate-load-plan
Trading

## Cargo load planner

Ce dépôt inclut un module de planification de chargement simplifié:

- Modélisation 3D de la soute et des ULD (taille, poids, priorité)
- Heuristique de bin-packing pour générer les positions et l'ordre de chargement
- Export du plan en format CSV ou XML
- API de simulation accessible via `POST /load-plan`

### Exemple rapide

```python
from load_planner import CargoHold, ULD, LoadPlanner

hold = CargoHold(length=10, width=5, height=3)
ulds = [
    ULD("A", length=2, width=2, height=2, weight=100, priority=1),
    ULD("B", length=3, width=2, height=1, weight=80, priority=0),
]
planner = LoadPlanner()
placements, unplaced = planner.plan_load(hold, ulds)
LoadPlanner.export_plan_csv(placements, "plan.csv")
```

### API

```bash
python load_planner_api.py
```

Puis envoyer une requête:

```bash
curl -X POST http://localhost:8000/load-plan \
     -H 'Content-Type: application/json' \
     -d '{
           "hold": {"length": 10, "width": 5, "height": 3},
           "ulds": [
             {"id": "A", "length": 2, "width": 2, "height": 2, "weight": 100, "priority": 1},
             {"id": "B", "length": 3, "width": 2, "height": 1, "weight": 80, "priority": 0}
           ]
         }'
```
=======
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
main
