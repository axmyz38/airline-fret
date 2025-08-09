"""Service REST basique pour fournir les prévisions de demande."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from .data_utils import load_data, add_season
from .model import train_models, predict
from .database import init_db, save_forecast

DATA_PATH = "demand_forecast/data/demand.csv"

class ForecastHandler(BaseHTTPRequestHandler):
    models = None
    db_conn = None

    @classmethod
    def setup_service(cls) -> None:
        data = load_data(DATA_PATH)
        add_season(data)
        cls.models = train_models(data)
        cls.db_conn = init_db()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/forecast":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return
        params = parse_qs(parsed.query)
        route = params.get("route", [None])[0]
        date = params.get("date", [None])[0]
        if not route or not date:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"route and date parameters are required")
            return
        try:
            forecast = predict(self.models, route, date)
        except Exception as exc:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(exc).encode())
            return
        save_forecast(self.db_conn, route, date, forecast)
        body = json.dumps({"route": route, "date": date, "forecast": forecast}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(port: int = 8000) -> None:
    ForecastHandler.setup_service()
    server = HTTPServer(("0.0.0.0", port), ForecastHandler)
    print(f"Serving on port {port}")
    server.serve_forever()

if __name__ == "__main__":
    run()
