from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import optuna
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

class EnsembleModel:
    """Meta-modèle pour combiner intelligemment les prévisions"""
    
    def __init__(self, base_models: Dict[str, BaseModel], config: Dict[str, Any]):
        self.base_models = base_models
        self.config = config
        self.meta_model = Ridge(alpha=1.0)
        self.weights = {}
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, use_stacking: bool = True) -> 'EnsembleModel':
        """Entraîne l'ensemble avec stacking ou blending"""
        
        if use_stacking:
            return self._fit_stacking(X, y)
        else:
            return self._fit_blending(X, y)
    
    def _fit_stacking(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """Stacking avec meta-modèle"""
        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = []
        meta_targets = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Prédictions des modèles de base sur validation
            val_predictions = {}
            
            # Parallélisation de l'entraînement
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {}
                
                for name, model in self.base_models.items():
                    future = executor.submit(self._train_and_predict, 
                                           model, X_train, y_train, X_val)
                    futures[future] = name
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        predictions = future.result()
                        val_predictions[name] = predictions
                    except Exception as e:
                        self.logger.error(f"Erreur modèle {name}: {e}")
            
            # Créer les features pour le meta-modèle
            if val_predictions:
                meta_X = pd.DataFrame(val_predictions, index=X_val.index)
                meta_features.append(meta_X)
                meta_targets.append(y_val)
        
        # Entraîner le meta-modèle
        if meta_features:
            X_meta = pd.concat(meta_features)
            y_meta = pd.concat(meta_targets)
            self.meta_model.fit(X_meta, y_meta)
            
            # Calculer les importances
            self.weights = dict(zip(val_predictions.keys(), 
                                  np.abs(self.meta_model.coef_) / np.sum(np.abs(self.meta_model.coef_))))
        
        return self
    
    def _fit_blending(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """Blending avec optimisation des poids"""
        
        def objective(trial):
            # Proposer des poids
            weights = {}
            remaining = 1.0
            
            for i, name in enumerate(self.base_models.keys()):
                if i < len(self.base_models) - 1:
                    w = trial.suggest_float(f'weight_{name}', 0, remaining)
                    weights[name] = w
                    remaining -= w
                else:
                    weights[name] = remaining
            
            # Calculer la performance avec ces poids
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Prédictions pondérées
                weighted_pred = np.zeros(len(y_val))
                
                for name, model in self.base_models.items():
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    weighted_pred += weights[name] * pred
                
                # Score (RMSE)
                rmse = np.sqrt(np.mean((weighted_pred - y_val.values) ** 2))
                scores.append(rmse)
            
            return np.mean(scores)
        
        # Optimisation Bayésienne
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        self.weights = study.best_params
        
        # Entraîner tous les modèles sur l'ensemble des données
        for name, model in self.base_models.items():
            model.fit(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Génère des prévisions d'ensemble"""
        
        if hasattr(self, 'meta_model') and self.meta_model is not None:
            # Stacking: utiliser le meta-modèle
            base_predictions = {}
            
            for name, model in self.base_models.items():
                base_predictions[name] = model.predict(X, horizon)
            
            # Créer les features pour le meta-modèle
            meta_X = pd.DataFrame(base_predictions)
            predictions = self.meta_model.predict(meta_X)
        else:
            # Blending: moyenne pondérée
            predictions = np.zeros(horizon)
            
            for name, model in self.base_models.items():
                weight = self.weights.get(name, 1.0 / len(self.base_models))
                predictions += weight * model.predict(X, horizon)
        
        return predictions
    
    def _train_and_predict(self, model, X_train, y_train, X_val):
        """Fonction helper pour la parallélisation"""
        model.fit(X_train, y_train)
        return model.predict(X_val)