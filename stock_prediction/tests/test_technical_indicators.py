import unittest
import numpy as np
import pandas as pd
from src.features.technical_indicators import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    """Tests unitaires pour les indicateurs techniques"""
    
    def setUp(self):
        """Prépare les données de test"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.data = pd.DataFrame({
            'AAPL.Open': np.random.randn(100).cumsum() + 100,
            'AAPL.High': np.random.randn(100).cumsum() + 102,
            'AAPL.Low': np.random.randn(100).cumsum() + 98,
            'AAPL.Close': np.random.randn(100).cumsum() + 100,
            'AAPL.Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_rsi_calculation(self):
        """Test le calcul du RSI"""
        close_prices = self.data['AAPL.Close'].values
        rsi = TechnicalIndicators._calculate_rsi_numba(close_prices)
        
        # Vérifier que RSI est entre 0 et 100
        self.assertTrue(np.all((rsi[~np.isnan(rsi)] >= 0) & (rsi[~np.isnan(rsi)] <= 100)))
        
        # Vérifier qu'il y a des NaN au début (période de warmup)
        self.assertTrue(np.isnan(rsi[:13]).all())
    
    def test_all_indicators(self):
        """Test le calcul de tous les indicateurs"""
        result = TechnicalIndicators.calculate_all_indicators(self.data, 'AAPL')
        
        # Vérifier que toutes les colonnes attendues sont présentes
        expected_cols = ['AAPL_MA5', 'AAPL_RSI', 'AAPL_MACD', 'AAPL_BB_Upper']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Vérifier les dimensions
        self.assertEqual(len(result), len(self.data))

if __name__ == '__main__':
    unittest.main()