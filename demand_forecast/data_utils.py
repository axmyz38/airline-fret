"""Utilities pour charger et préparer les données de demande."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

SEASONS = {
    1: "winter", 2: "winter", 12: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

def load_data(csv_path: str | Path) -> List[Dict[str, Any]]:
    """Charge les données depuis un fichier CSV."""
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def add_season(data: List[Dict[str, Any]]) -> None:
    """Ajoute une clé 'season' basée sur la date."""
    for row in data:
        date = datetime.strptime(row["date"], "%Y-%m-%d")
        row["season"] = SEASONS[date.month]

__all__ = ["load_data", "add_season"]
