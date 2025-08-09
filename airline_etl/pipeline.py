from pathlib import Path

from airline_etl.connectors.flights import import_flights
from airline_etl.connectors.aircraft import import_aircraft
from airline_etl.connectors.bookings import import_bookings
from airline_etl.db import init_db, insert_records

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DB_PATH = BASE_DIR / 'airline.db'


def run_pipeline():
    """Load datasets and store them into the relational database."""
    conn = init_db(DB_PATH)
    flights = import_flights(DATA_DIR / 'flights.csv')
    insert_records(conn, 'flights', flights)
    print(f"Loaded {len(flights)} flights")

    aircraft = import_aircraft(DATA_DIR / 'aircraft.csv')
    insert_records(conn, 'aircraft', aircraft)
    print(f"Loaded {len(aircraft)} aircraft records")

    bookings = import_bookings(DATA_DIR / 'bookings.csv')
    insert_records(conn, 'bookings', bookings)
    print(f"Loaded {len(bookings)} bookings")

    conn.close()


if __name__ == '__main__':
    run_pipeline()
