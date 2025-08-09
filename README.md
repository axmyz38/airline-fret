# trading-

Ce dépôt contient désormais un service simple de prévision de la demande.

## Démarrage rapide

```bash
python -m demand_forecast.api
```

Cela lance un petit serveur HTTP exposant l'endpoint :

```
GET /forecast?route=ROUTE&date=YYYY-MM-DD
```

Chaque requête renvoie une prévision basée sur les moyennes saisonnières
et l'enregistre dans la base `demand_forecast.db`.
