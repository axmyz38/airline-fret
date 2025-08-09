"""Modèle de prévision simplifié basé sur les moyennes saisonnières."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

from .data_utils import SEASONS

Model = Dict[str, Dict[str, float]]

def train_models(data: List[Dict[str, Any]]) -> Model:
    """Entraîne un modèle de moyenne saisonnière pour chaque route."""
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in data:
        route = row["route"]
        season = row["season"]
        demand = float(row["demand"])
        sums[route][season] += demand
        counts[route][season] += 1
    models: Model = {}
    for route, seasons in sums.items():
        models[route] = {}
        for season, total in seasons.items():
            models[route][season] = total / counts[route][season]
    return models

def predict(models: Model, route: str, date: str) -> float:
    """Prédit la demande pour une route et une date données."""
    dt = datetime.strptime(date, "%Y-%m-%d")
    season = SEASONS[dt.month]
    route_model = models.get(route)
    if not route_model:
        raise ValueError(f"Modèle inconnu pour la route {route}")
    season_values = route_model.get(season)
    if season_values is None:
        # moyenne globale de la route
        season_values = sum(route_model.values()) / len(route_model)
    return season_values

__all__ = ["train_models", "predict"]
