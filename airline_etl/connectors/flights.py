import csv
from pathlib import Path
from typing import List, Dict

from ..validation import normalize_date


def import_flights(source: Path) -> List[Dict[str, str]]:
    """Import flight information from a CSV file."""
    flights = []
    with source.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row['departure'] = normalize_date(row['departure'])
            row['arrival'] = normalize_date(row['arrival'])
            flights.append(row)
    return flights
