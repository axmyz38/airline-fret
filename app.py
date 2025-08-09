"""Minimal HTTP server exposing POST /plan for flight scheduling."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from typing import Tuple

from planner import plan_schedule, init_db, save_schedule


class PlanHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # pragma: no cover - network interaction
        if self.path != "/plan":
            self.send_error(404, "Not Found")
            return
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        schedule = plan_schedule(data["dates"], data["routes"], data["aircraft"])
        save_schedule(schedule)
        body = json.dumps({"schedule": schedule}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:  # pragma: no cover - entry point
    init_db()
    server = HTTPServer((host, port), PlanHandler)
    server.serve_forever()


if __name__ == "__main__":
    run()
