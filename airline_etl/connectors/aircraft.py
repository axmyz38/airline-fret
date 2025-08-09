import csv
from pathlib import Path
from typing import List, Dict

from ..validation import normalize_weight


def import_aircraft(source: Path) -> List[Dict[str, str]]:
    """Import aircraft capacities from a CSV file."""
    aircraft = []
    with source.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row['capacity'] = int(row['capacity'])
            row['max_weight_kg'] = normalize_weight(float(row['max_weight']), row['weight_unit'])
            del row['max_weight']
            del row['weight_unit']
            aircraft.append(row)
    return aircraft
