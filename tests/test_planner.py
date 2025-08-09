import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from planner import plan_schedule, init_db, save_schedule


def test_plan_and_save(tmp_path):
    db_path = tmp_path / "schedule.db"
    init_db(str(db_path))
    dates = ["2024-06-01"]
    routes = [
        {"id": "R1", "origin": "H1", "destination": "C1", "duration": 2, "demand": 100},
        {"id": "R2", "origin": "H1", "destination": "C2", "duration": 3, "demand": 120},
    ]
    aircraft = [
        {"id": "A1", "capacity": 200, "hub": "H1", "max_hours": 5, "maintenance": []},
        {"id": "A2", "capacity": 150, "hub": "H1", "max_hours": 5, "maintenance": []},
    ]
    schedule = plan_schedule(dates, routes, aircraft)
    assert len(schedule) == 2
    save_schedule(schedule, str(db_path))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT date, aircraft_id, route_id FROM flight_schedule").fetchall()
    assert len(rows) == 2
