import os
import sqlite3
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from demand_forecast import data_utils, model, database

DATA_CSV = "demand_forecast/data/demand.csv"

class TestDemandForecast(unittest.TestCase):
    def setUp(self):
        self.data = data_utils.load_data(DATA_CSV)
        data_utils.add_season(self.data)

    def test_train_and_predict(self):
        models = model.train_models(self.data)
        forecast = model.predict(models, "A-B", "2024-03-15")
        self.assertIsInstance(forecast, float)

    def test_save_forecast(self):
        conn = database.init_db(":memory:")
        database.save_forecast(conn, "A-B", "2024-03-15", 123.0)
        cur = conn.cursor()
        cur.execute("SELECT route, date, forecast FROM demand_forecast")
        row = cur.fetchone()
        self.assertEqual(row, ("A-B", "2024-03-15", 123.0))
        conn.close()

if __name__ == "__main__":
    unittest.main()
