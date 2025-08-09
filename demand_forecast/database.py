"""Gestion de la base SQLite pour les prévisions."""

import sqlite3
from datetime import datetime
from typing import Tuple

DB_PATH = "demand_forecast.db"


def init_db(path: str = DB_PATH) -> sqlite3.Connection:
    """Initialise la base de données et retourne la connexion."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS demand_forecast (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route TEXT NOT NULL,
            date TEXT NOT NULL,
            forecast REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn

def save_forecast(conn: sqlite3.Connection, route: str, date: str, forecast: float) -> None:
    """Enregistre une prévision dans la base de données."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO demand_forecast (route, date, forecast, created_at) VALUES (?, ?, ?, ?)",
        (route, date, forecast, datetime.utcnow().isoformat()),
    )
    conn.commit()

__all__ = ["init_db", "save_forecast", "DB_PATH"]
