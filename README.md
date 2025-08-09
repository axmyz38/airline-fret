# trading-

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
