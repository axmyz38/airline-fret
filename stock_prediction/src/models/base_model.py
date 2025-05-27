from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import pickle
import json
from datetime import datetime

class BaseModel(ABC):
    """Interface commune pour tous les modèles de prévision"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self.metrics = {}
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Entraîne le modèle"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Génère des prévisions"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Retourne les paramètres du modèle"""
        pass
    
    def save(self, path: str) -> None:
        """Sauvegarde le modèle"""
        with open(f"{path}/{self.__class__.__name__}.pkl", 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'metrics': self.metrics,
                'feature_names': self.feature_names
            }, f)
    
    def load(self, path: str) -> None:
        """Charge le modèle"""
        with open(f"{path}/{self.__class__.__name__}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.config = data['config']
            self.metrics = data['metrics']
            self.feature_names = data['feature_names']
            self.is_fitted = True
    
    def validate_prediction(self, predictions: np.ndarray, current_price: float) -> np.ndarray:
        """Valide et ajuste les prévisions pour éviter les valeurs aberrantes"""
        validated = predictions.copy()
        
        # Limiter les changements extrêmes
        max_daily_change = 0.1  # 10% max par jour
        
        for i in range(len(validated)):
            if i == 0:
                base_price = current_price
            else:
                base_price = validated[i-1]
            
            change_rate = (validated[i] - base_price) / base_price
            
            if abs(change_rate) > max_daily_change:
                validated[i] = base_price * (1 + np.sign(change_rate) * max_daily_change)
        
        return validated