import sqlite3
from pathlib import Path
from typing import Iterable, Dict

SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    flight_id TEXT PRIMARY KEY,
    aircraft_id TEXT,
    departure TEXT,
    arrival TEXT
);

CREATE TABLE IF NOT EXISTS aircraft (
    aircraft_id TEXT PRIMARY KEY,
    model TEXT,
    capacity INTEGER,
    max_weight_kg REAL
);

CREATE TABLE IF NOT EXISTS bookings (
    booking_id TEXT PRIMARY KEY,
    flight_id TEXT,
    passenger_name TEXT,
    seat TEXT,
    weight_kg REAL,
    FOREIGN KEY(flight_id) REFERENCES flights(flight_id)
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the database with the required schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_records(conn: sqlite3.Connection, table: str, records: Iterable[Dict]):
    """Insert records into a given table."""
    records = list(records)
    if not records:
        return
    keys = records[0].keys()
    placeholders = ",".join(["?"] * len(keys))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(keys)}) VALUES ({placeholders})"
    values = [tuple(r[k] for k in keys) for r in records]
    conn.executemany(sql, values)
    conn.commit()
