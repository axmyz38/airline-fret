"""Flask API exposing pricing endpoint and logging price decisions."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from flask import Flask, jsonify, request

from pricing_algorithm import emsr_b, choose_price

app = Flask(__name__)

# Default calibration parameters
PRICES = [100.0, 80.0, 60.0]
MEAN_DEMANDS = [20.0, 40.0, 60.0]
STD_DEVS = [5.0, 10.0, 15.0]
CAPACITY = 100

PROTECTION_LEVELS = emsr_b(PRICES, MEAN_DEMANDS, STD_DEVS)


def log_price(order_data: dict, price: float) -> None:
    """Persist price decisions into the pricing_history table."""
    conn = sqlite3.connect("pricing_history.db")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pricing_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            order_info TEXT,
            price REAL
        )
        """
    )
    cur.execute(
        "INSERT INTO pricing_history (timestamp, order_info, price) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), json.dumps(order_data), price),
    )
    conn.commit()
    conn.close()


@app.post("/price")
def price_endpoint():
    """Return the optimal price for a given order."""
    data = request.get_json(silent=True) or {}
    remaining = int(data.get("remaining_capacity", CAPACITY))
    price = choose_price(PRICES, PROTECTION_LEVELS, remaining)
    log_price(data, price)
    return jsonify({"price": price})


if __name__ == "__main__":
    app.run(debug=True)
