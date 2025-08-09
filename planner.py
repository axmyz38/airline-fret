import sqlite3
from itertools import permutations
from typing import List, Dict, Any

try:
    import pulp  # type: ignore
    HAS_PULP = True
except Exception:  # pragma: no cover - fall back when pulp is missing
    HAS_PULP = False


def plan_schedule(dates: List[str], routes: List[Dict[str, Any]], aircraft: List[Dict[str, Any]]):
    """Plan flights for given dates.

    If :mod:`pulp` is available, a MILP model is solved. Otherwise a simple
    backtracking search is used as a fallback so tests can run without the
    optional dependency.
    """
    if HAS_PULP:
        return _plan_with_pulp(dates, routes, aircraft)
    return _plan_with_backtracking(dates, routes, aircraft)


def _plan_with_pulp(dates, routes, aircraft):
    prob = pulp.LpProblem("flight_schedule", pulp.LpMinimize)
    x = {}
    for a in aircraft:
        for r in routes:
            for d in dates:
                x[(a["id"], r["id"], d)] = pulp.LpVariable(
                    f"x_{a['id']}_{r['id']}_{d}", cat="Binary"
                )
    prob += pulp.lpSum(x.values())
    for r in routes:
        for d in dates:
            prob += (
                pulp.lpSum(x[(a["id"], r["id"], d)] * a["capacity"] for a in aircraft)
                >= r["demand"],
                f"cap_{r['id']}_{d}",
            )
    for a in aircraft:
        for d in dates:
            if d in a.get("maintenance", []):
                for r in routes:
                    prob += x[(a["id"], r["id"], d)] == 0
            else:
                for r in routes:
                    if r["origin"] != a["hub"] or r["duration"] > a["max_hours"]:
                        prob += x[(a["id"], r["id"], d)] == 0
                prob += (
                    pulp.lpSum(x[(a["id"], r["id"], d)] for r in routes) <= 1,
                    f"one_route_{a['id']}_{d}",
                )
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    schedule = []
    for (a_id, r_id, d), var in x.items():
        if var.value() and var.value() > 0.5:
            schedule.append({"date": d, "aircraft_id": a_id, "route_id": r_id})
    return schedule


def _plan_with_backtracking(dates, routes, aircraft):
    schedule = []
    for d in dates:
        available = [a for a in aircraft if d not in a.get("maintenance", [])]
        day_assignments = []

        def backtrack(route_idx, used):
            if route_idx == len(routes):
                return True
            r = routes[route_idx]
            for a in available:
                if a["id"] in used:
                    continue
                if r["origin"] != a["hub"]:
                    continue
                if r["duration"] > a["max_hours"]:
                    continue
                if a["capacity"] < r["demand"]:
                    continue
                used.add(a["id"])
                day_assignments.append((a["id"], r["id"]))
                if backtrack(route_idx + 1, used):
                    return True
                used.remove(a["id"])
                day_assignments.pop()
            return False

        if not backtrack(0, set()):
            raise ValueError(f"No feasible schedule for date {d}")
        for a_id, r_id in day_assignments:
            schedule.append({"date": d, "aircraft_id": a_id, "route_id": r_id})
    return schedule


def init_db(db_path: str = "flight_schedule.db") -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS flight_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                aircraft_id TEXT,
                route_id TEXT
            )
            """
        )


def save_schedule(schedule, db_path: str = "flight_schedule.db") -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO flight_schedule (date, aircraft_id, route_id) VALUES (?, ?, ?)",
            [(s["date"], s["aircraft_id"], s["route_id"]) for s in schedule],
        )
