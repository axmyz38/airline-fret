import csv
from pathlib import Path
from typing import List, Dict

from ..validation import normalize_weight


def import_bookings(source: Path) -> List[Dict[str, str]]:
    """Import booking information from a CSV file."""
    bookings = []
    with source.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row['weight_kg'] = normalize_weight(float(row['weight']), row['weight_unit'])
            del row['weight']
            del row['weight_unit']
            bookings.append(row)
    return bookings
