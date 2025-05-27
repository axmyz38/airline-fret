import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional
import numba

class TechnicalIndicators:
    """Calcul vectorisé des indicateurs techniques"""
    
    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calcul RSI optimisé avec Numba"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period] = np.nan
        rsi[period] = 100. - 100. / (1. + rs)
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calcule tous les indicateurs techniques de manière vectorisée"""
        
        # Extraction des colonnes OHLCV
        open_col = f"{symbol}.Open"
        high_col = f"{symbol}.High"
        low_col = f"{symbol}.Low"
        close_col = f"{symbol}.Close"
        volume_col = f"{symbol}.Volume"
        
        # Vérification de l'existence des colonnes
        required_cols = [open_col, high_col, low_col, close_col, volume_col]
        if not all(col in df.columns for col in required_cols):
            # Gérer le cas MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                open_vals = df[(symbol, 'Open')].values
                high_vals = df[(symbol, 'High')].values
                low_vals = df[(symbol, 'Low')].values
                close_vals = df[(symbol, 'Close')].values
                volume_vals = df[(symbol, 'Volume')].values
            else:
                raise ValueError(f"Colonnes manquantes pour {symbol}")
        else:
            open_vals = df[open_col].values
            high_vals = df[high_col].values
            low_vals = df[low_col].values
            close_vals = df[close_col].values
            volume_vals = df[volume_col].values
        
        result = pd.DataFrame(index=df.index)
        
        # Moyennes mobiles
        result[f'{symbol}_MA5'] = talib.SMA(close_vals, timeperiod=5)
        result[f'{symbol}_MA10'] = talib.SMA(close_vals, timeperiod=10)
        result[f'{symbol}_MA20'] = talib.SMA(close_vals, timeperiod=20)
        result[f'{symbol}_MA50'] = talib.SMA(close_vals, timeperiod=50)
        result[f'{symbol}_MA200'] = talib.SMA(close_vals, timeperiod=200)
        
        # EMA
        result[f'{symbol}_EMA12'] = talib.EMA(close_vals, timeperiod=12)
        result[f'{symbol}_EMA26'] = talib.EMA(close_vals, timeperiod=26)
        
        # RSI optimisé
        result[f'{symbol}_RSI'] = TechnicalIndicators._calculate_rsi_numba(close_vals)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_vals)
        result[f'{symbol}_MACD'] = macd
        result[f'{symbol}_MACD_Signal'] = macd_signal
        result[f'{symbol}_MACD_Hist'] = macd_hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close_vals)
        result[f'{symbol}_BB_Upper'] = upper
        result[f'{symbol}_BB_Middle'] = middle
        result[f'{symbol}_BB_Lower'] = lower
        result[f'{symbol}_BB_Width'] = upper - lower
        result[f'{symbol}_BB_Position'] = (close_vals - lower) / (upper - lower)
        
        # Stochastic
        slowk, slowd = talib.STOCH(high_vals, low_vals, close_vals)
        result[f'{symbol}_STOCH_K'] = slowk
        result[f'{symbol}_STOCH_D'] = slowd
        
        # ADX
        result[f'{symbol}_ADX'] = talib.ADX(high_vals, low_vals, close_vals)
        
        # ATR
        result[f'{symbol}_ATR'] = talib.ATR(high_vals, low_vals, close_vals)
        
        # CCI
        result[f'{symbol}_CCI'] = talib.CCI(high_vals, low_vals, close_vals)
        
        # Williams %R
        result[f'{symbol}_WILLR'] = talib.WILLR(high_vals, low_vals, close_vals)
        
        # OBV vectorisé
        obv = np.zeros_like(close_vals)
        obv[0] = volume_vals[0]
        
        price_diff = np.diff(close_vals)
        volume_direction = np.where(price_diff > 0, volume_vals[1:], 
                                   np.where(price_diff < 0, -volume_vals[1:], 0))
        obv[1:] = np.cumsum(volume_direction)
        result[f'{symbol}_OBV'] = obv
        
        # Volume indicators
        result[f'{symbol}_Volume_MA20'] = talib.SMA(volume_vals, timeperiod=20)
        result[f'{symbol}_Volume_Ratio'] = volume_vals / result[f'{symbol}_Volume_MA20']
        
        # Price patterns
        result[f'{symbol}_DOJI'] = talib.CDLDOJI(open_vals, high_vals, low_vals, close_vals)
        result[f'{symbol}_HAMMER'] = talib.CDLHAMMER(open_vals, high_vals, low_vals, close_vals)
        result[f'{symbol}_ENGULFING'] = talib.CDLENGULFING(open_vals, high_vals, low_vals, close_vals)
        
        # Support/Resistance levels using rolling min/max
        result[f'{symbol}_Support_20'] = pd.Series(low_vals).rolling(20).min()
        result[f'{symbol}_Resistance_20'] = pd.Series(high_vals).rolling(20).max()
        
        # Volatility measures
        result[f'{symbol}_Volatility_20'] = pd.Series(close_vals).pct_change().rolling(20).std()
        result[f'{symbol}_Volatility_Ratio'] = result[f'{symbol}_Volatility_20'] / result[f'{symbol}_Volatility_20'].rolling(50).mean()
        
        # Calendar features
        result[f'{symbol}_DayOfWeek'] = df.index.dayofweek
        result[f'{symbol}_MonthEnd'] = df.index.is_month_end.astype(int)
        result[f'{symbol}_QuarterEnd'] = df.index.is_quarter_end.astype(int)
        
        return result