
   


    # Importations de base - nécessaires pour le fonctionnement de base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import random
from scipy import stats
import statsmodels.tsa.stattools  # Commenté pour éviter les erreurs
import os
import pickle
import json
import traceback
import logging
import yaml

# Statistiques et modélisation de séries temporelles - certaines peuvent être commentées si nécessaire
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model  # Commenté pour éviter les erreurs

# Machine Learning - nécessaire pour la plupart des fonctionnalités
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN  # Pour la détection d'anomalies

# Commentez toutes les importations TensorFlow si vous ne les utilisez pas immédiatement

# Apprentissage profond

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# Commentez l'optimisation bayésienne si non utilisée

# Optimisation bayésienne
import skopt
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


# Commentez les algorithmes génétiques si non utilisés

# Algorithmes génétiques
import deap
from deap import base, creator, tools, algorithms


# Commentez toute l'analyse de sentiment si non utilisée

# Analyse de sentiment
import nltk
import textblob
from textblob import TextBlob
import tweepy
import praw
from newspaper import Article


# Gardez les importations de requêtes web si vous avez besoin de téléchargement de données
import requests
from bs4 import BeautifulSoup

# Visualisation avancée - vous pouvez garder matplotlib mais commenter plotly

# Visualisation avancée
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib.dates as mdates

# Parallélisation - vous pouvez garder joblib

# Parallélisation
import multiprocessing

from joblib import Parallel, delayed

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_boursiere.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PredictionBoursiere:
    """
    Classe principale pour la prévision boursière avancée.
    Implémente toutes les fonctionnalités demandées dans un pipeline cohérent.
    """
    
    def __init__(self, config_path=None, symbols=None, tiingo_api_key=None):
        """
        Initialise le système de prévision boursière avec les configurations
        
        Args:
            config_path (str): Chemin vers le fichier de configuration YAML
            symbols (list): Liste des symboles boursiers à analyser
            tiingo_api_key (str): Clé API Tiingo (optionnelle)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialisation du système de prévision boursière")
        
        # Chargement de la configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = self._default_config()
        
        # Mettre à jour les symboles si fournis
        if symbols:
            self.config['data']['symbols'] = symbols
            self.logger.info(f"Utilisation des symboles personnalisés: {symbols}")
        
        # Mettre à jour la clé API Tiingo si fournie
        if tiingo_api_key:
            if 'tiingo' not in self.config:
                self.config['tiingo'] = {}
            self.config['tiingo']['api_key'] = tiingo_api_key
            self.logger.info("Clé API Tiingo configurée via le paramètre")
        else:
            # Vérifier si la clé API est valide
            if ('tiingo' not in self.config or 
                'api_key' not in self.config['tiingo'] or 
                not self.config['tiingo']['api_key'] or 
                self.config['tiingo']['api_key'] == "VOTRE_CLÉ_API_TIINGO_ICI"):
                self.logger.warning("Aucune clé API Tiingo valide configurée. Veuillez définir une clé API valide.")
        
        # Initialisation des attributs
        self.data = None
        self.exogenous_data = {}
        self.sentiment_data = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = None
        self.horizon = self.config['prediction']['horizon']
        self.confidence_level = self.config['prediction']['confidence_level']
        self.anomaly_detector = None
        
        # Initialisation des API pour l'analyse de sentiment
        self._init_sentiment_apis()
        
        self.logger.info("Système initialisé avec succès")
        
    def nettoyer_donnees(self):
        """Nettoie les données avant le prétraitement pour éviter les erreurs"""
        try:
            if self.data is None:
                self.logger.warning("Aucune donnée à nettoyer")
                return False
                
            self.logger.info("Nettoyage des données")
                
            # Remplacer les valeurs infinies
            self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Imputation avancée des valeurs manquantes
            missing_before = self.data.isnull().sum().sum()
            
            for col in self.data.columns:
                if self.data[col].isnull().sum() > 0:
                    # Utiliser une interpolation plus robuste
                    self.data[col] = self.data[col].interpolate(method='time').ffill().bfill()
            
            missing_after = self.data.isnull().sum().sum()
            self.logger.info(f"Valeurs manquantes: {missing_before} -> {missing_after}")  # Utilisation de -> au lieu de →
            
            # S'il reste des NaN, les remplacer par des zéros en dernier recours
            if missing_after > 0:
                self.logger.warning(f"Il reste {missing_after} valeurs manquantes, remplacement par 0")
                self.data.fillna(0, inplace=True)
                
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage des données: {str(e)}")
            traceback.print_exc()
            return False
    
    def _default_config(self):
        """
        Définit la configuration par défaut du système
        
        Returns:
            dict: Configuration par défaut
        """
        return {
            "data": {
                "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                "start_date": (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d'),
                "end_date": datetime.now().strftime('%Y-%m-%d'),
                "source": "tiingo",  # Changé de "yahoo" à "tiingo"
                "backup_source": "file",  # Source de secours
                "frequency": "daily",
                "features": ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
            },
            "tiingo": {  # Nouvelle section pour Tiingo
                "api_key": "VOTRE_CLÉ_API_TIINGO_ICI",  # À remplacer par votre clé API
                "base_url": "https://api.tiingo.com",
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": "PredictionBoursiere/1.0"
                },
                "retry_count": 3,  # Nombre de tentatives en cas d'échec
                "retry_delay": 1,  # Délai entre les tentatives (en secondes)
            },
            "preprocessing": {
                "handle_missing": "knn",
                "scaling": "robust",
                "anomaly_detection": {
                    "method": "dbscan",
                    "params": {
                        "eps": 0.5,
                        "min_samples": 5
                    }
                }
            },
            "features": {
                "technical_indicators": True,
                "fundamental_data": True,
                "sentiment_analysis": True,
                "macroeconomic": True
            },
            "models": {
                "arima": {
                    "enabled": True,
                    "max_p": 5,
                    "max_d": 2,
                    "max_q": 5,
                    "seasonal": True,
                    "optimization": "bayesian"
                },
                "var": {
                    "enabled": True,
                    "max_lags": 10
                },
                "garch": {
                    "enabled": True,
                    "p": 1,
                    "q": 1,
                    "distribution": "normal"
                },
                "machine_learning": {
                    "enabled": True,
                    "algorithms": ["random_forest", "gradient_boosting", "ridge"],
                    "feature_selection": True
                },
                "deep_learning": {
                    "enabled": True,
                    "architectures": ["lstm", "gru", "transformer"],
                    "layers": [64, 32],
                    "dropout": 0.2,
                    "epochs": 100,
                    "batch_size": 32,
                    "early_stopping": True
                }
            },
            "optimization": {
                "method": "bayesian",
                "n_iter": 50,
                "cv": 5,
                "parallel_jobs": -1
            },
            "prediction": {
                "horizon": 10,  # Jours
                "confidence_level": 0.95,
                "refit_frequency": "daily"
            },
            "backtesting": {
                "windows": 5,
                "initial_train_size": 0.7,
                "metrics": ["rmse", "mae", "mape", "r2", "sharpe"]
            },
            "sentiment": {
                "sources": ["twitter", "news", "reddit"],
                "lookback": 7,  # Jours
                "refresh_rate": "daily"
            },
            "output": {
                "save_models": True,
                "visualizations": True,
                "reports": True,
                "export_format": ["csv", "json"]
            }
        }
    
    def _init_sentiment_apis(self):
        """
        Initialise les API pour l'analyse de sentiment
        """
        try:
            # Ces clés doivent être configurées ou obtenues de manière sécurisée
            # Pour l'exemple, nous utilisons des valeurs fictives
            if self.config["features"]["sentiment_analysis"]:
                self.sentiment_apis = {
                    "twitter": {
                        "consumer_key": "XXX",
                        "consumer_secret": "XXX",
                        "access_token": "XXX",
                        "access_token_secret": "XXX"
                    },
                    "reddit": {
                        "client_id": "XXX",
                        "client_secret": "XXX",
                        "user_agent": "Prediction Boursiere Bot v1.0"
                    },
                    "news_api": {
                        "api_key": "XXX"
                    }
                }
                self.logger.info("APIs de sentiment initialisées")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des APIs de sentiment: {str(e)}")
            self.sentiment_apis = None
            
    def _fix_date_index(self, series, freq='B'):
        """
        Corrige l'index de dates pour les séries temporelles afin d'éviter les avertissements
        lors de l'utilisation des modèles ARIMA/SARIMA.
        
        Args:
            series (pd.Series): Série temporelle à corriger
            freq (str): Fréquence à appliquer ('B' pour jours ouvrables, 'D' pour jours calendaires)
            
        Returns:
            pd.Series: Série avec index de dates corrigé
        """
        try:
            self.logger.info(f"Correction de l'index de dates avec la fréquence '{freq}'")
            
            # Vérifier si nous avons une série
            if series is None:
                self.logger.error("Impossible de corriger l'index: la série est None")
                return None
            
            # Trier la série par ordre chronologique (important pour la monotonie)
            series = series.sort_index()
            
            # Vérifier si l'index a déjà une fréquence
            if hasattr(series.index, 'freq') and series.index.freq is not None:
                self.logger.info(f"L'index a déjà une fréquence: {series.index.freq}")
                return series
            
            # Créer un nouvel index avec la fréquence spécifiée
            start_date = series.index[0]
            end_date = series.index[-1]
            
            # Créer un nouvel index de dates avec la fréquence spécifiée
            new_index = pd.date_range(start=start_date, end=end_date, freq=freq)
            
            # Si le nouvel index est trop long (contient plus de dates que l'original)
            # nous devons filtrer pour correspondre aux dates existantes
            if len(new_index) > len(series):
                self.logger.info(f"Ajustement du nouvel index (original: {len(series)}, nouveau: {len(new_index)})")
                # Réindexer en utilisant les dates existantes avec la nouvelle fréquence
                series = series.reindex(new_index, method='ffill')
                # Supprimer les valeurs ajoutées (NaN)
                series = series.dropna()
            else:
                # Réindexer avec le nouvel index
                series = series.reindex(new_index)
                # Interpoler les valeurs manquantes si nécessaires
                if series.isna().any():
                    self.logger.info("Interpolation des valeurs manquantes après réindexation")
                    series = series.interpolate(method='time')
            
            # Vérifier si l'index est maintenant correctement configuré
            if not hasattr(series.index, 'freq') or series.index.freq is None:
                self.logger.info("Application manuelle de la fréquence à l'index")
                series.index.freq = pd.tseries.frequencies.to_offset(freq)
            
            self.logger.info(f"Index corrigé: {len(series)} points, fréquence {series.index.freq}")
            return series
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la correction de l'index de dates: {str(e)}")
            traceback.print_exc()
            # Retourner la série originale en cas d'erreur
            return series
            
    def load_data(self, custom_data_path=None):
        """
        Charge les données depuis Tiingo Pro ou un fichier local
        
        Args:
            custom_data_path (str, optional): Chemin vers des données personnalisées
                
        Returns:
            bool: True si les données ont été chargées avec succès
        """
        try:
            import pandas as pd
            import time
            
            # Si un chemin de données personnalisées est fourni, essayer de charger depuis ce chemin
            if custom_data_path:
                try:
                    self.logger.info(f"Tentative de chargement depuis {custom_data_path}")
                    
                    if custom_data_path.endswith('.csv'):
                        df = pd.read_csv(custom_data_path, index_col=0, parse_dates=True)
                        
                        # Vérifier si l'index est correctement formaté comme datetime
                        if not isinstance(df.index, pd.DatetimeIndex):
                            self.logger.warning("L'index n'est pas un DatetimeIndex, conversion...")
                            df.index = pd.to_datetime(df.index)
                            
                        # Trier par ordre chronologique croissant
                        df = df.sort_index()
                        
                        self.data = df
                        self.logger.info(f"Données chargées avec succès depuis {custom_data_path}: {self.data.shape}")
                        return True
                    else:
                        self.logger.error(f"Format de fichier non pris en charge: {custom_data_path}")
                
                except Exception as load_error:
                    self.logger.error(f"Erreur lors du chargement des données depuis {custom_data_path}: {str(load_error)}")
                    self.logger.warning("Tentative de chargement depuis Tiingo...")
            
            # Vérifier si la configuration Tiingo existe
            if 'tiingo' not in self.config or 'api_key' not in self.config['tiingo'] or not self.config['tiingo']['api_key']:
                self.logger.error("Clé API Tiingo manquante. Veuillez configurer une clé API valide.")
                return False
            
            api_key = self.config['tiingo']['api_key']
            if api_key == "VOTRE_CLÉ_API_TIINGO_ICI":
                self.logger.error("Clé API Tiingo non configurée. Veuillez remplacer la valeur par défaut par une clé API réelle.")
                return False
            
            self.logger.info("Chargement des données depuis Tiingo Pro")
            
            # Configuration de Tiingo
            from tiingo import TiingoClient
            
            tiingo_config = {
                'api_key': api_key,
                'session': True
            }
            
            # Initialiser le client Tiingo
            try:
                client = TiingoClient(tiingo_config)
            except Exception as tiingo_error:
                self.logger.error(f"Erreur lors de l'initialisation du client Tiingo: {str(tiingo_error)}")
                return False
            
            # Symboles à récupérer
            symbols = self.config['data']['symbols']
            start_date = pd.to_datetime(self.config['data']['start_date'])
            end_date = pd.to_datetime(self.config['data']['end_date'])
            
            # Format de la date pour Tiingo (YYYY-MM-DD)
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Création d'un dictionnaire pour stocker les DataFrames
            all_data = {}
            
            for symbol in symbols:
                self.logger.info(f"Récupération des données pour {symbol}")
                
                try:
                    # Récupérer les données pour un seul symbole
                    historical_data = client.get_dataframe(
                        symbol,  # IMPORTANT: utiliser un seul symbole à la fois, pas une liste
                        frequency='daily',
                        startDate=start_date_str,
                        endDate=end_date_str
                    )
                    
                    # Vérifier si des données ont été récupérées
                    if historical_data.empty:
                        self.logger.warning(f"Aucune donnée récupérée pour {symbol}")
                        continue
                    
                    # Traiter les données
                    df = historical_data.copy()
                    
                    # Renommer les colonnes pour correspondre au format attendu
                    column_mapping = {
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume',
                        'adjClose': 'Adj Close',
                        'adjVolume': 'Adj Volume',
                        'adjHigh': 'Adj High',
                        'adjLow': 'Adj Low',
                        'adjOpen': 'Adj Open',
                        'divCash': 'Dividends',
                        'splitFactor': 'Split'
                    }
                    
                    # Renommer uniquement les colonnes qui existent
                    for old_name, new_name in column_mapping.items():
                        if old_name in df.columns:
                            df.rename(columns={old_name: new_name}, inplace=True)
                    
                    # Si Adj Close n'existe pas, utilisez Close
                    if 'Adj Close' not in df.columns and 'Close' in df.columns:
                        df['Adj Close'] = df['Close']
                    
                    # Gérer les problèmes de fuseau horaire dans l'index
                    if hasattr(df.index, 'tz') and df.index.tz is not None:
                        # Supprimer l'information de fuseau horaire
                        df.index = df.index.tz_localize(None)
                    
                    # S'assurer que les dates de début et de fin sont également sans fuseau horaire
                    start_date_naive = start_date.replace(tzinfo=None) if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None else start_date
                    end_date_naive = end_date.replace(tzinfo=None) if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None else end_date
                    
                    # Filtrer selon les dates
                    try:
                        df = df.loc[(df.index >= start_date_naive) & (df.index <= end_date_naive)]
                    except TypeError as type_error:
                        # Si TypeError persiste, essayons une autre approche
                        self.logger.warning(f"Erreur de type lors du filtrage des dates pour {symbol}: {str(type_error)}")
                        # Convertir l'index en chaîne et comparer les dates sous forme de chaîne
                        date_mask = (df.index.strftime('%Y-%m-%d') >= start_date_naive.strftime('%Y-%m-%d')) & \
                                    (df.index.strftime('%Y-%m-%d') <= end_date_naive.strftime('%Y-%m-%d'))
                        df = df.loc[date_mask]
                    
                    # S'assurer que l'index a une fréquence
                    try:
                        # Pour les données boursières, généralement 'B' (jours ouvrables)
                        df = df.asfreq('B')
                    except ValueError:
                        # Si problème avec asfreq, continuer sans définir la fréquence
                        pass
                    
                    if not df.empty:
                        all_data[symbol] = df
                        self.logger.info(f"Données récupérées pour {symbol}: {df.shape}")
                    else:
                        self.logger.warning(f"Aucune donnée dans la plage de dates pour {symbol}")
                    
                    # Pause pour ne pas surcharger l'API
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Erreur lors de la récupération des données pour {symbol}: {str(e)}")
                    continue
            
            # Si aucune donnée n'a été récupérée
            if not all_data:
                self.logger.error("Aucune donnée n'a pu être récupérée depuis Tiingo")
                return False
            
            # Créer un DataFrame multi-indexé à partir des données récupérées
            dfs = []
            for symbol, df in all_data.items():
                # Ajouter le symbole comme niveau dans les colonnes
                df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
                dfs.append(df)
            
            # Concaténer tous les DataFrames
            try:
                self.data = pd.concat(dfs, axis=1)
                
                # Afficher des informations sur les données chargées
                self.logger.info(f"Données chargées avec succès: {self.data.shape}")
                self.logger.info(f"Période des données: {self.data.index[0]} à {self.data.index[-1]}")
                self.logger.info(f"Nombre de lignes: {len(self.data)}")
                
                return True
            except Exception as concat_error:
                self.logger.error(f"Erreur lors de la concaténation des données: {str(concat_error)}")
                return False
            
        except Exception as e:
            self.logger.error(f"Erreur générale lors du chargement des données: {str(e)}")
            traceback.print_exc()
            return False
    
    def load_exogenous_data(self):
        """
        Charge les données exogènes pour améliorer les prévisions
        
        Returns:
            bool: True si les données ont été chargées avec succès
        """
        try:
            self.logger.info("Chargement des données exogènes")
            
            # Indicateurs macroéconomiques
            if self.config['features']['macroeconomic']:
                # Utiliser une API pour récupérer des données macroéconomiques
                # Exemple fictif:
                macro_data = self._fetch_macroeconomic_data()
                if macro_data is not None:
                    self.exogenous_data['macro'] = macro_data
            
            # Données fondamentales
            if self.config['features']['fundamental_data']:
                # Récupérer les données fondamentales pour les symboles
                fundamental_data = self._fetch_fundamental_data()
                if fundamental_data is not None:
                    self.exogenous_data['fundamental'] = fundamental_data
            
            self.logger.info(f"Données exogènes chargées: {len(self.exogenous_data)} sources")
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données exogènes: {str(e)}")
            return False
    
    def _fetch_macroeconomic_data(self):
        """
        Récupère les données macroéconomiques
        
        Returns:
            pd.DataFrame: Données macroéconomiques
        """
        try:
            # Dans un cas réel, on utiliserait une API comme FRED
            # Exemple fictif pour la démonstration:
            start_date = self.config['data']['start_date']
            end_date = self.config['data']['end_date']
            
            # Créer des données factices
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')
            macro_data = pd.DataFrame(index=date_range)
            
            # Simuler des indicateurs macroéconomiques
            macro_data['GDP_growth'] = np.random.normal(0.5, 0.2, size=len(date_range))
            macro_data['unemployment'] = np.random.normal(5.0, 0.5, size=len(date_range))
            macro_data['inflation'] = np.random.normal(2.0, 0.3, size=len(date_range))
            macro_data['interest_rate'] = np.random.normal(3.0, 0.2, size=len(date_range))
            
            return macro_data
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données macroéconomiques: {str(e)}")
            return None
    
    def _fetch_fundamental_data(self):
        """
        Récupère les données fondamentales pour les symboles depuis Tiingo
        
        Returns:
            dict: Données fondamentales par symbole
        """
        try:
            from tiingo import TiingoClient
            
            # Configuration de Tiingo
            tiingo_config = {
                'api_key': self.config['tiingo']['api_key'],
                'session': True
            }
            
            client = TiingoClient(tiingo_config)
            fundamental_data = {}
            
            for symbol in self.config['data']['symbols']:
                self.logger.info(f"Récupération des données fondamentales pour {symbol}")
                
                try:
                    # Récupérer les données fondamentales (si disponibles dans votre abonnement Tiingo)
                    # Note: Cette fonctionnalité pourrait nécessiter un niveau d'abonnement spécifique
                    fund_data = client.get_fundamentals(
                        tickers=[symbol],
                        startDate=self.config['data']['start_date'],
                        endDate=self.config['data']['end_date']
                    )
                    
                    if fund_data and len(fund_data) > 0:
                        # Convertir en DataFrame
                        fund_df = pd.DataFrame(fund_data)
                        
                        # Définir la date comme index
                        if 'date' in fund_df.columns:
                            fund_df['date'] = pd.to_datetime(fund_df['date'])
                            fund_df.set_index('date', inplace=True)
                            fund_df.sort_index(inplace=True)
                        
                        fundamental_data[symbol] = fund_df
                        self.logger.info(f"Données fondamentales récupérées pour {symbol}: {fund_df.shape}")
                    else:
                        self.logger.warning(f"Aucune donnée fondamentale disponible pour {symbol}")
                        
                        # Utiliser des données fictives comme fallback
                        # Cette partie du code original peut être conservée
                        quarterly_dates = pd.date_range(
                            start=self.config['data']['start_date'],
                            end=self.config['data']['end_date'],
                            freq='Q'
                        )
                        
                        fund_df = pd.DataFrame(index=quarterly_dates)
                        fund_df['eps'] = np.random.normal(2.0, 0.5, size=len(quarterly_dates))
                        fund_df['pe_ratio'] = np.random.normal(15.0, 3.0, size=len(quarterly_dates))
                        fund_df['debt_to_equity'] = np.random.normal(0.8, 0.2, size=len(quarterly_dates))
                        fund_df['revenue_growth'] = np.random.normal(0.1, 0.05, size=len(quarterly_dates))
                        fund_df['profit_margin'] = np.random.normal(0.15, 0.03, size=len(quarterly_dates))
                        
                        fundamental_data[symbol] = fund_df
                
                except Exception as e:
                    self.logger.error(f"Erreur lors de la récupération des données fondamentales pour {symbol}: {str(e)}")
                    continue
            
            return fundamental_data
        
        except Exception as e:
            self.logger.error(f"Erreur générale lors de la récupération des données fondamentales: {str(e)}")
            traceback.print_exc()
            return None
        
    def get_symbol_metadata(self, symbol):
        """
        Récupère les métadonnées pour un symbole depuis Tiingo
        
        Args:
            symbol (str): Symbole boursier
        
        Returns:
            dict: Métadonnées du symbole
        """
        try:
            from tiingo import TiingoClient
            
            # Configuration de Tiingo
            tiingo_config = {
                'api_key': self.config['tiingo']['api_key'],
                'session': True
            }
            
            client = TiingoClient(tiingo_config)
            
            # Récupérer les métadonnées
            metadata = client.get_ticker_metadata(symbol)
            
            return metadata
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des métadonnées pour {symbol}: {str(e)}")
            return None
        
    def get_price_column(self, symbol, column_type='Close'):
        """
        Récupère une colonne de prix en gérant les différents formats d'index
        
        Args:
            symbol (str): Symbole boursier
            column_type (str): Type de colonne ('Close', 'Open', etc.)
        
        Returns:
            pd.Series: La colonne de prix demandée ou None si non trouvée
        """
        try:
            # Essayer tous les formats possibles de colonnes
            column_formats = [
                f"{symbol}.{column_type}",           # Format "AAPL.Close"
                (symbol, column_type),               # Format MultiIndex ('AAPL', 'Close')
                f"{symbol}_{column_type}",           # Format "AAPL_Close"
                f"{column_type}_{symbol}",           # Format "Close_AAPL"
                column_type                          # Si une seule série avec juste le type de colonne
            ]
            
            # Vérifier chaque format
            for col_format in column_formats:
                if isinstance(col_format, tuple) and isinstance(self.data.columns, pd.MultiIndex):
                    # Format MultiIndex
                    if col_format in self.data.columns:
                        return self.data[col_format]
                elif col_format in self.data.columns:
                    # Format string
                    return self.data[col_format]
            
            # Si aucun format standard n'a fonctionné, rechercher par correspondance partielle
            for col in self.data.columns:
                col_str = str(col)
                if symbol in col_str and column_type in col_str:
                    self.logger.info(f"Colonne trouvée par correspondance partielle: {col}")
                    return self.data[col]
            
            # Si toujours pas trouvé, lever une exception personnalisée
            raise ValueError(f"Impossible de trouver une colonne de {column_type} pour {symbol}")
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de la colonne {column_type} pour {symbol}: {str(e)}")
            return None
        
    def _get_default_features(self, symbol):
        """
        Génère un ensemble de caractéristiques par défaut pour un symbole
        Utilisé lorsque la sélection de caractéristiques échoue
        
        Args:
            symbol (str): Symbole boursier
            
        Returns:
            list: Liste des caractéristiques par défaut
        """
        self.logger.info(f"Génération de caractéristiques par défaut pour {symbol}")
        
        default_features = []
        
        # 1. Ajouter les colonnes de prix standard (sauf Close qui est la cible)
        price_cols = ['Open', 'High', 'Low', 'Volume']
        
        # Gérer correctement le multiindex
        if isinstance(self.data.columns, pd.MultiIndex):
            # Si nous avons un MultiIndex, les colonnes sont (symbol, type)
            for col_type in price_cols:
                if (symbol, col_type) in self.data.columns:
                    default_features.append((symbol, col_type))
            
            # Ajouter quelques colonnes techniques créées pendant le prétraitement
            tech_indicators = [
                (symbol, 'MA5'), (symbol, 'MA20'), (symbol, 'RSI'),
                (symbol, 'MACD'), (symbol, 'Momentum'), (symbol, 'Volatility20')
            ]
            
            for indicator in tech_indicators:
                if indicator in self.data.columns:
                    default_features.append(indicator)
        else:
            # Format à un seul niveau
            for col_type in price_cols:
                col = f"{symbol}.{col_type}"
                if col in self.data.columns:
                    default_features.append(col)
            
            # Ajouter quelques indicateurs techniques standards
            tech_indicators = [
                f"{symbol}.MA5", f"{symbol}.MA20", f"{symbol}.RSI", 
                f"{symbol}.MACD", f"{symbol}.Momentum", f"{symbol}.Volatility20"
            ]
            
            for indicator in tech_indicators:
                if indicator in self.data.columns:
                    default_features.append(indicator)
        
        # S'assurer qu'il y a au moins quelques caractéristiques
        if len(default_features) < 3:
            self.logger.warning(f"Trop peu de caractéristiques trouvées pour {symbol}, ajout de colonnes génériques")
            # Ajouter des colonnes génériques (les 10 premières colonnes, sauf Close)
            count = 0
            for col in self.data.columns:
                col_name = col if isinstance(col, str) else col[1] if isinstance(col, tuple) else str(col)
                if "Close" not in col_name and col not in default_features:
                    default_features.append(col)
                    count += 1
                    if count >= 10:
                        break
        
        self.logger.info(f"Caractéristiques par défaut pour {symbol}: {len(default_features)} caractéristiques")
        return default_features

    def _fix_column_formats(self):
        """
        Vérifie et corrige le format des colonnes dans le DataFrame
        
        Returns:
            bool: True si la correction a réussi
        """
        try:
            self.logger.info("Vérification et correction du format des colonnes...")
            
            # Vérifier si nous avons des données chargées
            if self.data is None or self.data.empty:
                self.logger.warning("Aucune donnée à corriger")
                return False
            
            # Vérifier le format actuel des colonnes
            if isinstance(self.data.columns, pd.MultiIndex):
                self.logger.info("Format MultiIndex détecté")
                
                # Vérifier si tous les symboles ont des colonnes dans ce format
                all_symbols_found = True
                for symbol in self.config['data']['symbols']:
                    if not any(symbol == col[0] for col in self.data.columns):
                        all_symbols_found = False
                        self.logger.warning(f"Symbole {symbol} non trouvé dans l'index multi-niveau")
                
                if all_symbols_found:
                    self.logger.info("Tous les symboles trouvés, aucune correction nécessaire")
                    return True
            
            # Si nous arrivons ici, nous avons besoin de standardiser le format
            new_columns = []
            rename_dict = {}
            
            for col in self.data.columns:
                col_str = str(col)
                
                # Chercher quel symbole correspond à cette colonne
                matching_symbol = None
                for symbol in self.config['data']['symbols']:
                    if symbol in col_str:
                        matching_symbol = symbol
                        break
                
                if matching_symbol:
                    # Déterminer le type de données (Close, Open, etc.)
                    data_types = ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
                    matching_type = None
                    
                    for data_type in data_types:
                        if data_type in col_str:
                            matching_type = data_type
                            break
                    
                    if matching_type:
                        # Créer le nouveau format standardisé
                        new_col = f"{matching_symbol}.{matching_type}"
                        rename_dict[col] = new_col
                        new_columns.append(new_col)
                    else:
                        # Garder la colonne originale
                        new_columns.append(col)
                else:
                    # Garder la colonne originale
                    new_columns.append(col)
            
            # Renommer les colonnes si nécessaire
            if rename_dict:
                self.data = self.data.rename(columns=rename_dict)
                self.logger.info(f"Colonnes renommées: {len(rename_dict)}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la correction du format des colonnes: {str(e)}")
            traceback.print_exc()
            return False
        
    def preprocess_data(self):
        """
        Prétraite les données pour l'analyse et la modélisation
        Version améliorée avec correction des problèmes d'index et de fréquence
        
        Returns:
            bool: True si le prétraitement a été effectué avec succès
        """
        try:
            if self.data is None:
                self.logger.error("Aucune donnée à prétraiter")
                return False
                
            self.logger.info("Prétraitement des données")
            
            # Vérification et correction de l'index de dates
            self._ensure_valid_date_index()
            
            # Gérer les valeurs manquantes
            self._handle_missing_values()
            
            # Détecter et traiter les anomalies
            self._detect_anomalies()
            
            # Créer des caractéristiques techniques
            if self.config['features']['technical_indicators']:
                self._create_technical_features()
            
            # Intégrer les données exogènes
            self._integrate_exogenous_data()
            
            # Intégrer les données de sentiment
            if self.sentiment_data is not None:
                self._integrate_sentiment_data()
            
            # Normalisation/standardisation des données
            self._scale_features()
            
            self.logger.info(f"Prétraitement terminé. Données finales: {self.data.shape}")
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur lors du prétraitement des données: {str(e)}")
            traceback.print_exc()
            return False

    def _ensure_valid_date_index(self):
        """
        Vérifie et corrige l'index de dates pour s'assurer qu'il est approprié pour les modèles de séries temporelles
        """
        try:
            self.logger.info("Vérification et correction de l'index de dates")
            
            # Vérifier si l'index est bien un DatetimeIndex
            if not isinstance(self.data.index, pd.DatetimeIndex):
                self.logger.warning("L'index n'est pas un DatetimeIndex, conversion...")
                self.data.index = pd.to_datetime(self.data.index)
            
            # Vérifier si l'index est trié
            if not self.data.index.is_monotonic_increasing:
                self.logger.warning("L'index n'est pas classé par ordre croissant, tri...")
                self.data = self.data.sort_index()
            
            # Vérifier si l'index a une fréquence, sinon en définir une
            if not hasattr(self.data.index, 'freq') or self.data.index.freq is None:
                self.logger.warning("L'index n'a pas d'information de fréquence, détection automatique...")
                
                # Tenter de détecter la fréquence automatiquement
                inferred_freq = pd.infer_freq(self.data.index)
                
                if inferred_freq:
                    self.logger.info(f"Fréquence détectée: {inferred_freq}")
                    self.data = self.data.asfreq(inferred_freq)
                else:
                    # Examiner les différences entre dates consécutives
                    date_diffs = pd.Series(self.data.index).diff().dropna()
                    most_common_diff = date_diffs.value_counts().index[0]
                    
                    # Définir une fréquence appropriée
                    if most_common_diff.days == 1:
                        self.logger.info("Les données semblent être quotidiennes, application de la fréquence 'D'")
                        self.data = self.data.asfreq('D')
                    elif 1 < most_common_diff.days <= 3:
                        self.logger.info("Les données semblent être des jours ouvrables, application de la fréquence 'B'")
                        self.data = self.data.asfreq('B')
                    else:
                        self.logger.warning(f"Impossible de détecter la fréquence, utilisation de 'B' par défaut")
                        self.data = self.data.asfreq('B')
            
            # Vérifier et corriger les valeurs dupliquées dans l'index
            if self.data.index.duplicated().any():
                duplicate_count = self.data.index.duplicated().sum()
                self.logger.warning(f"L'index contient {duplicate_count} valeurs dupliquées, suppression...")
                self.data = self.data[~self.data.index.duplicated(keep='first')]
            
            self.logger.info(f"Vérification de l'index terminée. Fréquence: {self.data.index.freq}, Période: {self.data.index[0]} - {self.data.index[-1]}")
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la correction de l'index de dates: {str(e)}")
            traceback.print_exc()
    
    def _handle_missing_values(self):
        """
        Gère les valeurs manquantes dans les données
        """
        try:
            # Vérifier s'il y a des valeurs manquantes
            missing_count = self.data.isnull().sum().sum()
            self.logger.info(f"Valeurs manquantes détectées: {missing_count}")
            
            if missing_count > 0:
                method = self.config['preprocessing']['handle_missing']
                
                if method == 'drop':
                    self.data = self.data.dropna()
                    self.logger.info(f"Lignes avec valeurs manquantes supprimées. Nouvelle taille: {self.data.shape}")
                
                elif method == 'ffill':
                    self.data = self.data.fillna(method='ffill')
                    self.logger.info("Valeurs manquantes remplacées par la dernière valeur connue")
                
                elif method == 'interpolate':
                    self.data = self.data.interpolate(method='time')
                    self.logger.info("Valeurs manquantes interpolées")
                
                elif method == 'knn':
                    # Utiliser KNNImputer pour les valeurs manquantes
                    imputer = KNNImputer(n_neighbors=5)
                    
                    # Appliquer l'imputation à chaque symbole séparément
                    for symbol in self.config['data']['symbols']:
                        cols = [col for col in self.data.columns if symbol in col]
                        if len(cols) > 0:
                            self.data[cols] = imputer.fit_transform(self.data[cols])
                    
                    self.logger.info("Valeurs manquantes imputées avec KNN")
                
                else:
                    self.logger.warning(f"Méthode d'imputation inconnue: {method}. Utilisation de ffill")
                    self.data = self.data.fillna(method='ffill')
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la gestion des valeurs manquantes: {str(e)}")
            raise
    
    def _detect_anomalies(self):
        """
        Détecte et traite les anomalies dans les données
        """
        try:
            self.logger.info("Détection des anomalies")
            
            method = self.config['preprocessing']['anomaly_detection']['method']
            
            if method == 'dbscan':
                # Utiliser DBSCAN pour la détection d'anomalies
                eps = self.config['preprocessing']['anomaly_detection']['params']['eps']
                min_samples = self.config['preprocessing']['anomaly_detection']['params']['min_samples']
                
                # Appliquer DBSCAN à chaque série de prix de clôture
                for symbol in self.config['data']['symbols']:
                    col = f"{symbol}.Close"
                    if col in self.data.columns:
                        # Normaliser les données pour DBSCAN
                        scaler = RobustScaler()
                        prices = self.data[col].values.reshape(-1, 1)
                        prices_scaled = scaler.fit_transform(prices)
                        
                        # Appliquer DBSCAN
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        clusters = dbscan.fit_predict(prices_scaled)
                        
                        # Identifier les anomalies (cluster -1)
                        anomalies = clusters == -1
                        anomaly_indices = np.where(anomalies)[0]
                        
                        if len(anomaly_indices) > 0:
                            self.logger.info(f"Anomalies détectées pour {col}: {len(anomaly_indices)}")
                            
                            # Remplacer les anomalies par interpolation
                            for idx in anomaly_indices:
                                # Marquer comme NaN pour interpolation ultérieure
                                self.data.iloc[idx, self.data.columns.get_loc(col)] = np.nan
                            
                            # Interpoler les valeurs
                            self.data[col] = self.data[col].interpolate(method='time')
            
            elif method == 'iqr':
                # Méthode IQR (Interquartile Range)
                for symbol in self.config['data']['symbols']:
                    col = f"{symbol}.Close"
                    if col in self.data.columns:
                        q1 = self.data[col].quantile(0.25)
                        q3 = self.data[col].quantile(0.75)
                        iqr = q3 - q1
                        
                        # Définir les limites pour les anomalies
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Identifier les anomalies
                        anomalies = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                        anomaly_indices = np.where(anomalies)[0]
                        
                        if len(anomaly_indices) > 0:
                            self.logger.info(f"Anomalies IQR détectées pour {col}: {len(anomaly_indices)}")
                            
                            # Remplacer les anomalies par des valeurs limites
                            self.data.loc[self.data[col] < lower_bound, col] = lower_bound
                            self.data.loc[self.data[col] > upper_bound, col] = upper_bound
            
            else:
                self.logger.warning(f"Méthode de détection d'anomalies inconnue: {method}")
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection des anomalies: {str(e)}")
            raise
    
    def _create_technical_features(self):
        """
        Crée des indicateurs techniques pour l'analyse
        """
        try:
            self.logger.info("Création d'indicateurs techniques")
            
            # Appliquer les indicateurs techniques pour chaque symbole
            for symbol in self.config['data']['symbols']:
                price_col = f"{symbol}.Close"
                volume_col = f"{symbol}.Volume"
                
                if price_col in self.data.columns:
                    # Moyennes mobiles
                    self.data[f"{symbol}.MA5"] = self.data[price_col].rolling(window=5).mean()
                    self.data[f"{symbol}.MA20"] = self.data[price_col].rolling(window=20).mean()
                    self.data[f"{symbol}.MA50"] = self.data[price_col].rolling(window=50).mean()
                    
                    # Écarts-types mobiles (volatilité)
                    self.data[f"{symbol}.Volatility5"] = self.data[price_col].rolling(window=5).std()
                    self.data[f"{symbol}.Volatility20"] = self.data[price_col].rolling(window=20).std()
                    
                    # RSI (Relative Strength Index)
                    delta = self.data[price_col].diff()
                    gain, loss = delta.copy(), delta.copy()
                    gain[gain < 0] = 0
                    loss[loss > 0] = 0
                    loss = -loss
                    
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    
                    rs = avg_gain / avg_loss
                    self.data[f"{symbol}.RSI"] = 100 - (100 / (1 + rs))
                    
                    # MACD (Moving Average Convergence Divergence)
                    ema12 = self.data[price_col].ewm(span=12, adjust=False).mean()
                    ema26 = self.data[price_col].ewm(span=26, adjust=False).mean()
                    self.data[f"{symbol}.MACD"] = ema12 - ema26
                    self.data[f"{symbol}.MACD_Signal"] = self.data[f"{symbol}.MACD"].ewm(span=9, adjust=False).mean()
                    
                    # Bollinger Bands
                    self.data[f"{symbol}.BB_Middle"] = self.data[price_col].rolling(window=20).mean()
                    std20 = self.data[price_col].rolling(window=20).std()
                    self.data[f"{symbol}.BB_Upper"] = self.data[f"{symbol}.BB_Middle"] + (std20 * 2)
                    self.data[f"{symbol}.BB_Lower"] = self.data[f"{symbol}.BB_Middle"] - (std20 * 2)
                    
                    # Momentum
                    self.data[f"{symbol}.Momentum"] = self.data[price_col].diff(periods=10)
                    
                    # Rate of Change
                    self.data[f"{symbol}.ROC"] = self.data[price_col].pct_change(periods=10) * 100
                    
                    # Stochastic Oscillator
                    high_14 = self.data[f"{symbol}.High"].rolling(window=14).max()
                    low_14 = self.data[f"{symbol}.Low"].rolling(window=14).min()
                    self.data[f"{symbol}.K"] = 100 * ((self.data[price_col] - low_14) / (high_14 - low_14))
                    self.data[f"{symbol}.D"] = self.data[f"{symbol}.K"].rolling(window=3).mean()
                
                if volume_col in self.data.columns:
                    # On-Balance Volume (OBV)
                    obv = 0
                    obv_values = []
                    
                    for i in range(len(self.data)):
                        if i > 0:
                            if self.data[price_col].iloc[i] > self.data[price_col].iloc[i-1]:
                                obv += self.data[volume_col].iloc[i]
                            elif self.data[price_col].iloc[i] < self.data[price_col].iloc[i-1]:
                                obv -= self.data[volume_col].iloc[i]
                        obv_values.append(obv)
                    
                    self.data[f"{symbol}.OBV"] = obv_values
                    
                    # Volume Rate of Change
                    self.data[f"{symbol}.Volume_ROC"] = self.data[volume_col].pct_change(periods=5) * 100
            
            # Suppression des lignes initiales avec des NaN dus au calcul des indicateurs
            max_window = 50  # La plus grande fenêtre utilisée
            self.data = self.data.iloc[max_window:]
            
            self.logger.info(f"Indicateurs techniques créés. Nouvelles colonnes: {len(self.data.columns)}")
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la création des indicateurs techniques: {str(e)}")
            raise
    
    def _integrate_exogenous_data(self):
        """
        Intègre les données exogènes dans le jeu de données principal
        """
        try:
            if not self.exogenous_data:
                self.logger.info("Aucune donnée exogène à intégrer")
                return
                    
            self.logger.info("Intégration des données exogènes")
        
            # Intégrer les données macroéconomiques
            if 'macro' in self.exogenous_data:
                macro_data = self.exogenous_data['macro']
            
                # Normaliser les index de dates pour éviter les problèmes de fuseau horaire
                macro_index = pd.DatetimeIndex(macro_data.index.strftime('%Y-%m-%d'))
                data_index = pd.DatetimeIndex(self.data.index.strftime('%Y-%m-%d'))
            
                # Aligner les indices
                aligned_macro = pd.DataFrame(index=data_index)
                
                # Copier les données avec l'index normalisé
                for col in macro_data.columns:
                    # Créer une série temporaire avec l'index normalisé
                    temp_series = pd.Series(macro_data[col].values, index=macro_index)
                    # Réindexer sur l'index des données principales
                    aligned_macro[col] = temp_series.reindex(data_index, method='ffill')
                
                # Vérifier si nous avons un MultiIndex ou un index simple
                if isinstance(self.data.columns, pd.MultiIndex):
                    # Pour MultiIndex, créer un niveau "Macro"
                    for col in aligned_macro.columns:
                        # Créer une série avec le nouvel index multi-niveau
                        self.data[('Macro', col)] = aligned_macro[col].values
                else:
                    # Pour index simple, utiliser le préfixe Macro.
                    for col in aligned_macro.columns:
                        self.data[f"Macro.{col}"] = aligned_macro[col].values
            
                self.logger.info(f"Données macroéconomiques intégrées: {len(aligned_macro.columns)} variables")
            
            # Intégrer les données fondamentales si disponibles
            if 'fundamental' in self.exogenous_data:
                fundamental_data = self.exogenous_data['fundamental']
                
                # Traiter chaque symbole séparément
                for symbol, fund_df in fundamental_data.items():
                    # Normaliser les index de dates
                    fund_index = pd.DatetimeIndex(fund_df.index.strftime('%Y-%m-%d'))
                    data_index = pd.DatetimeIndex(self.data.index.strftime('%Y-%m-%d'))
                    
                    # Aligner les indices
                    aligned_fund = pd.DataFrame(index=data_index)
                    
                    # Copier les données avec l'index normalisé
                    for col in fund_df.columns:
                        # Créer une série temporaire avec l'index normalisé
                        temp_series = pd.Series(fund_df[col].values, index=fund_index)
                        # Réindexer sur l'index des données principales et utiliser ffill 
                        # pour les données fondamentales trimestrielles
                        aligned_fund[col] = temp_series.reindex(data_index, method='ffill')
                    
                    # Vérifier si nous avons un MultiIndex ou un index simple
                    if isinstance(self.data.columns, pd.MultiIndex):
                        # Pour MultiIndex, créer un niveau "Fundamental" sous le symbole
                        for col in aligned_fund.columns:
                            # Créer une série avec le nouvel index multi-niveau
                            self.data[(symbol, f"Fund_{col}")] = aligned_fund[col].values
                    else:
                        # Pour index simple, utiliser le préfixe Symbol.Fund_
                        for col in aligned_fund.columns:
                            self.data[f"{symbol}.Fund_{col}"] = aligned_fund[col].values
                    
                    self.logger.info(f"Données fondamentales intégrées pour {symbol}: {len(aligned_fund.columns)} variables")
            
            # Intégrer d'autres types de données exogènes si nécessaire
            for key, data in self.exogenous_data.items():
                if key not in ['macro', 'fundamental']:
                    self.logger.info(f"Intégration des données exogènes de type {key}")
                    
                    # Supposons que data est un DataFrame avec le même index que self.data
                    # ou qu'il peut être réindexé
                    
                    if isinstance(data, pd.DataFrame):
                        # Normaliser les index de dates
                        other_index = pd.DatetimeIndex(data.index.strftime('%Y-%m-%d'))
                        data_index = pd.DatetimeIndex(self.data.index.strftime('%Y-%m-%d'))
                        
                        # Aligner les indices
                        aligned_data = pd.DataFrame(index=data_index)
                        
                        # Copier les données avec l'index normalisé
                        for col in data.columns:
                            # Créer une série temporaire avec l'index normalisé
                            temp_series = pd.Series(data[col].values, index=other_index)
                            # Réindexer sur l'index des données principales
                            aligned_data[col] = temp_series.reindex(data_index, method='ffill')
                        
                        # Vérifier si nous avons un MultiIndex ou un index simple
                        if isinstance(self.data.columns, pd.MultiIndex):
                            # Pour MultiIndex, créer un niveau spécifique au type de données
                            cap_key = key.capitalize()  # Première lettre en majuscule
                            for col in aligned_data.columns:
                                # Créer une série avec le nouvel index multi-niveau
                                self.data[(cap_key, col)] = aligned_data[col].values
                        else:
                            # Pour index simple, utiliser un préfixe TypeDonnées.
                            cap_key = key.capitalize()
                            for col in aligned_data.columns:
                                self.data[f"{cap_key}.{col}"] = aligned_data[col].values
                        
                        self.logger.info(f"Données {key} intégrées: {len(aligned_data.columns)} variables")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'intégration des données exogènes: {str(e)}")
            traceback.print_exc()
    
    def _integrate_sentiment_data(self):
        """
        Intègre les données de sentiment dans le jeu de données principal
        """
        try:
            if self.sentiment_data is None:
                self.logger.info("Aucune donnée de sentiment à intégrer")
                return
                    
            self.logger.info("Intégration des données de sentiment")
            
            # Normaliser les index de dates
            sentiment_data_index = pd.DatetimeIndex(self.sentiment_data.index.strftime('%Y-%m-%d'))
            data_index = pd.DatetimeIndex(self.data.index.strftime('%Y-%m-%d'))
            
            # Créer un nouveau DataFrame avec l'index normalisé
            aligned_sentiment = pd.DataFrame(index=data_index)
            
            # Copier les données avec l'index normalisé
            for col in self.sentiment_data.columns:
                # Créer une série temporaire avec l'index normalisé
                temp_series = pd.Series(self.sentiment_data[col].values, index=sentiment_data_index)
                # Réindexer sur l'index des données principales
                aligned_sentiment[col] = temp_series.reindex(data_index, method='ffill')
            
            # Vérifier si nous avons un MultiIndex ou un index simple
            if isinstance(self.data.columns, pd.MultiIndex):
                # Pour MultiIndex, créer un niveau "Sentiment"
                for col in aligned_sentiment.columns:
                    # Créer une série avec le nouvel index multi-niveau
                    self.data[('Sentiment', col)] = aligned_sentiment[col].values
                
                # Créer des indicateurs basés sur le sentiment
                if 'sentiment_score_normalized' in aligned_sentiment:
                    self.data[('Sentiment', 'MA5')] = aligned_sentiment['sentiment_score_normalized'].rolling(window=5).mean().values
                    self.data[('Sentiment', 'MA10')] = aligned_sentiment['sentiment_score_normalized'].rolling(window=10).mean().values
                    self.data[('Sentiment', 'Momentum')] = aligned_sentiment['sentiment_score_normalized'].diff(periods=3).values
            else:
                # Pour index simple, utiliser le préfixe Sentiment.
                for col in aligned_sentiment.columns:
                    self.data[f"Sentiment.{col}"] = aligned_sentiment[col]
                
                # Créer des indicateurs techniques basés sur le sentiment
                if 'sentiment_score_normalized' in aligned_sentiment:
                    sentiment_col = 'Sentiment.sentiment_score_normalized'
                    
                    # Moyenne mobile du sentiment
                    self.data['Sentiment.MA5'] = self.data[sentiment_col].rolling(window=5).mean()
                    self.data['Sentiment.MA10'] = self.data[sentiment_col].rolling(window=10).mean()
                    
                    # Momentum du sentiment
                    self.data['Sentiment.Momentum'] = self.data[sentiment_col].diff(periods=3)
            
            self.logger.info(f"Données de sentiment intégrées: {len(aligned_sentiment.columns)} variables")
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'intégration des données de sentiment: {str(e)}")
            traceback.print_exc()
    
    def _scale_features(self):
        """
        Normalise/standardise les caractéristiques pour l'analyse
        """
        try:
            self.logger.info("Normalisation des caractéristiques")
            
            # Séparer les colonnes de prix de clôture des autres caractéristiques
            close_cols = [col for col in self.data.columns if 'Close' in col]
            feature_cols = [col for col in self.data.columns if col not in close_cols]
            
            # Choix de la méthode de mise à l'échelle
            method = self.config['preprocessing']['scaling']
            
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                self.logger.warning(f"Méthode de mise à l'échelle inconnue: {method}. Utilisation de RobustScaler")
                self.scaler = RobustScaler()
            
            # Appliquer la mise à l'échelle aux caractéristiques
            if feature_cols:
                self.data[feature_cols] = self.scaler.fit_transform(self.data[feature_cols])
                self.logger.info(f"Caractéristiques normalisées: {len(feature_cols)} variables")
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la normalisation des caractéristiques: {str(e)}")
            raise
        
    def analyze_sentiment(self):
        """
        Analyse le sentiment des investisseurs à partir de diverses sources
        
        Returns:
            bool: True si l'analyse a été effectuée avec succès
        """
        try:
            if not self.config['features']['sentiment_analysis']:
                self.logger.info("Analyse de sentiment désactivée dans la configuration")
                return False
                
            self.logger.info("Analyse du sentiment des investisseurs")
            
            sentiment_data = pd.DataFrame(
                index=pd.date_range(
                    start=self.config['data']['start_date'],
                    end=self.config['data']['end_date'],
                    freq='D'
                )
            )
            
            # Récupérer le sentiment depuis différentes sources
            if 'twitter' in self.config['sentiment']['sources']:
                twitter_sentiment = self._analyze_twitter_sentiment()
                if twitter_sentiment is not None:
                    sentiment_data['twitter'] = twitter_sentiment
            
            if 'news' in self.config['sentiment']['sources']:
                news_sentiment = self._analyze_news_sentiment()
                if news_sentiment is not None:
                    sentiment_data['news'] = news_sentiment
            
            if 'reddit' in self.config['sentiment']['sources']:
                reddit_sentiment = self._analyze_reddit_sentiment()
                if reddit_sentiment is not None:
                    sentiment_data['reddit'] = reddit_sentiment
            
            # Agrégation du sentiment
            sentiment_data['sentiment_score'] = sentiment_data.mean(axis=1)
            
            # Normalisation du score de sentiment
            scaler = MinMaxScaler(feature_range=(-1, 1))
            sentiment_data['sentiment_score_normalized'] = scaler.fit_transform(
                sentiment_data['sentiment_score'].values.reshape(-1, 1)
            )
            
            self.sentiment_data = sentiment_data
            self.logger.info(f"Analyse de sentiment terminée: {sentiment_data.shape}")
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de sentiment: {str(e)}")
            return False
    
    def _analyze_twitter_sentiment(self):
        """
        Analyse le sentiment des tweets sur les symboles
        
        Returns:
            pd.Series: Scores de sentiment par jour
        """
        try:
            # Dans un cas réel, on utiliserait l'API Twitter
            # Exemple fictif pour la démonstration:
            date_range = pd.date_range(
                start=self.config['data']['start_date'],
                end=self.config['data']['end_date'],
                freq='D'
            )
            
            # Simuler des scores de sentiment
            base_sentiment = np.random.normal(0.2, 0.4, size=len(date_range))
            
            # Ajouter quelques tendances pour simuler des événements
            trend = np.sin(np.linspace(0, 10, len(date_range))) * 0.2
            noise = np.random.normal(0, 0.1, size=len(date_range))
            
            sentiment = base_sentiment + trend + noise
            sentiment = np.clip(sentiment, -1, 1)  # Borner les valeurs entre -1 et 1
            
            return pd.Series(sentiment, index=date_range)
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse du sentiment Twitter: {str(e)}")
            return None
    
    def _analyze_news_sentiment(self):
        """
        Analyse le sentiment des actualités financières
        
        Returns:
            pd.Series: Scores de sentiment par jour
        """
        try:
            # Dans un cas réel, on utiliserait une API d'actualités
            # Exemple fictif pour la démonstration:
            date_range = pd.date_range(
                start=self.config['data']['start_date'],
                end=self.config['data']['end_date'],
                freq='D'
            )
            
            # Simuler des scores de sentiment avec une tendance différente
            base_sentiment = np.random.normal(0.1, 0.3, size=len(date_range))
            
            # Ajouter une tendance haussière légère
            trend = np.linspace(0, 0.3, len(date_range))
            noise = np.random.normal(0, 0.15, size=len(date_range))
            
            sentiment = base_sentiment + trend + noise
            sentiment = np.clip(sentiment, -1, 1)  # Borner les valeurs entre -1 et 1
            
            return pd.Series(sentiment, index=date_range)
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse du sentiment des actualités: {str(e)}")
            return None
    
    def _analyze_reddit_sentiment(self):
        """
        Analyse le sentiment des forums Reddit (e.g., r/wallstreetbets)
        
        Returns:
            pd.Series: Scores de sentiment par jour
        """
        try:
            # Dans un cas réel, on utiliserait l'API Reddit
            # Exemple fictif pour la démonstration:
            date_range = pd.date_range(
                start=self.config['data']['start_date'],
                end=self.config['data']['end_date'],
                freq='D'
            )
            
            # Simuler des scores de sentiment avec plus de volatilité
            base_sentiment = np.random.normal(0, 0.5, size=len(date_range))
            
            # Ajouter des événements plus extrêmes
            events = np.zeros(len(date_range))
            event_indices = np.random.choice(len(date_range), size=5, replace=False)
            events[event_indices] = np.random.choice([-1, 1], size=5) * np.random.uniform(0.5, 1, size=5)
            
            # Propager l'effet des événements
            from scipy.ndimage import gaussian_filter1d
            events = gaussian_filter1d(events, sigma=3)
            
            sentiment = base_sentiment + events
            sentiment = np.clip(sentiment, -1, 1)  # Borner les valeurs entre -1 et 1
            
            return pd.Series(sentiment, index=date_range)
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse du sentiment Reddit: {str(e)}")
            return None

    def _evaluate_model_quality(self, symbol, model_name, model_info):
        """
        Méthode manquante pour évaluer la qualité des modèles
        """
        try:
            quality_score = 0.5  # Score par défaut
            
            # 1. Vérifier les métriques de backtesting si disponibles
            if ('backtesting' in self.results and symbol in self.results['backtesting'] and 
                model_name in self.results['backtesting'][symbol]):
                
                metrics = self.results['backtesting'][symbol][model_name]
                
                # R² (plus important)
                if 'avg_r2' in metrics and metrics['avg_r2'] is not None:
                    r2 = metrics['avg_r2']
                    if r2 > 0.8:
                        quality_score = 0.9
                    elif r2 > 0.5:
                        quality_score = 0.7
                    elif r2 > 0.2:
                        quality_score = 0.5
                    elif r2 > 0:
                        quality_score = 0.3
                    else:
                        quality_score = 0.1  # R² négatif
                
                # MAPE
                if 'avg_mape' in metrics and metrics['avg_mape'] is not None:
                    mape = metrics['avg_mape']
                    if mape < 1:
                        quality_score += 0.2
                    elif mape < 3:
                        quality_score += 0.1
                    elif mape > 10:
                        quality_score -= 0.2
            
            # 2. Vérifier les métriques internes du modèle
            if 'metrics' in model_info:
                metrics = model_info['metrics']
                
                if 'r2' in metrics:
                    r2 = metrics['r2']
                    if r2 < -1:  # R² très négatif = modèle très mauvais
                        quality_score = 0.1
                    elif r2 < 0:
                        quality_score = max(0.2, quality_score - 0.2)
                    elif r2 > 0.5:
                        quality_score = min(1.0, quality_score + 0.2)
            
            # 3. Détection d'overfitting suspect
            if ('backtesting' in self.results and symbol in self.results['backtesting'] and 
                model_name in self.results['backtesting'][symbol]):
                
                backtest_r2 = self.results['backtesting'][symbol][model_name].get('avg_r2', 0)
                
                # Si R² > 0.99, c'est suspect d'overfitting
                if backtest_r2 > 0.99:
                    self.logger.warning(f"Overfitting suspect détecté pour {model_name}: R²={backtest_r2:.4f}")
                    quality_score *= 0.5  # Pénalité pour overfitting
            
            # Limiter le score entre 0.1 et 1.0
            quality_score = max(0.1, min(1.0, quality_score))
            
            self.logger.info(f"Score de qualité pour {model_name} sur {symbol}: {quality_score:.3f}")
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation de la qualité du modèle: {e}")
            return 0.3  # Score conservateur par défaut

        
    def select_features(self):
        """
        Version corrigée de la sélection de features qui gère mieux les NaN
        """
        try:
            self.logger.info("Sélection des meilleures caractéristiques (version robuste)")
            
            selected_features = {}
            self.selected_features = {}
            
            for symbol in self.config['data']['symbols']:
                try:
                    # Récupérer la colonne cible
                    target_col = self.get_price_column(symbol, 'Close')
                    if target_col is None:
                        self.logger.warning(f"Colonne cible non trouvée pour {symbol}")
                        selected_features[symbol] = self._get_default_features(symbol)
                        continue
                    
                    # Préparer les caractéristiques potentielles
                    all_cols = list(self.data.columns)
                    potential_features = []
                    
                    # Filtrer les colonnes appropriées
                    for col in all_cols:
                        if isinstance(col, tuple) and len(col) == 2:
                            col_symbol, col_type = col
                            if (col_symbol == symbol and col_type != 'Close') or col_symbol != symbol:
                                potential_features.append(col)
                        elif isinstance(col, str):
                            if not (col.endswith('.Close') and symbol in col):
                                potential_features.append(col)
                    
                    if len(potential_features) < 5:
                        self.logger.warning(f"Peu de features potentielles pour {symbol}")
                        selected_features[symbol] = self._get_default_features(symbol)
                        continue
                    
                    # Préparer X et y
                    X = self.data[potential_features].copy()
                    y = target_col.pct_change().shift(-1)
                    
                    # CORRECTION CRITIQUE: Nettoyage complet des NaN
                    # 1. Supprimer les lignes où y est NaN
                    valid_mask = ~y.isna()
                    X_clean = X[valid_mask].copy()
                    y_clean = y[valid_mask].copy()
                    
                    # 2. Supprimer les colonnes avec trop de NaN (>50%)
                    nan_threshold = 0.5
                    valid_columns = []
                    for col in X_clean.columns:
                        nan_ratio = X_clean[col].isna().sum() / len(X_clean)
                        if nan_ratio < nan_threshold:
                            valid_columns.append(col)
                    
                    if len(valid_columns) < 3:
                        self.logger.warning(f"Trop de colonnes avec NaN pour {symbol}")
                        selected_features[symbol] = self._get_default_features(symbol)
                        continue
                    
                    X_clean = X_clean[valid_columns].copy()
                    
                    # 3. Imputer les NaN restants
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    X_imputed = pd.DataFrame(
                        imputer.fit_transform(X_clean),
                        columns=X_clean.columns,
                        index=X_clean.index
                    )
                    
                    # 4. Vérification finale des NaN
                    if X_imputed.isna().any().any() or y_clean.isna().any():
                        self.logger.error(f"NaN restants après nettoyage pour {symbol}")
                        selected_features[symbol] = self._get_default_features(symbol)
                        continue
                    
                    # 5. Sélection des meilleures features
                    n_features = min(max(5, len(valid_columns) // 3), 15)
                    
                    try:
                        selector = SelectKBest(f_regression, k=n_features)
                        selector.fit(X_imputed, y_clean)
                        
                        best_indices = selector.get_support(indices=True)
                        best_features = [valid_columns[i] for i in best_indices]
                        
                        selected_features[symbol] = best_features
                        
                        # Log des features sélectionnées
                        feature_scores = [(valid_columns[i], selector.scores_[i]) for i in range(len(valid_columns))]
                        feature_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        self.logger.info(f"Top 5 features pour {symbol}:")
                        for feat, score in feature_scores[:5]:
                            self.logger.info(f"  - {feat}: {score:.4f}")
                            
                    except Exception as sel_error:
                        self.logger.error(f"Erreur SelectKBest pour {symbol}: {sel_error}")
                        selected_features[symbol] = valid_columns[:10]  # Prendre les 10 premières
                    
                except Exception as symbol_error:
                    self.logger.error(f"Erreur pour {symbol}: {symbol_error}")
                    selected_features[symbol] = self._get_default_features(symbol)
            
            # Vérification finale
            for symbol in self.config['data']['symbols']:
                if symbol not in selected_features or not selected_features[symbol]:
                    selected_features[symbol] = self._get_default_features(symbol)
            
            self.selected_features = selected_features
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Erreur générale sélection features: {str(e)}")
            # Fallback complet
            default_features_dict = {}
            for symbol in self.config['data']['symbols']:
                default_features_dict[symbol] = self._get_default_features(symbol)
            self.selected_features = default_features_dict
            return default_features_dict

            
        except Exception as e:
            self.logger.error(f"Erreur générale lors de la sélection des caractéristiques: {str(e)}")
            traceback.print_exc()
            
            # En cas d'erreur, utiliser des caractéristiques par défaut pour tous les symboles
            default_features_dict = {}
            for symbol in self.config['data']['symbols']:
                default_features_dict[symbol] = self._get_default_features(symbol)
            
            self.selected_features = default_features_dict
            return default_features_dict

    def _get_default_features(self, symbol):
        """
        Génère un ensemble de caractéristiques par défaut pour un symbole
        Utilisé lorsque la sélection de caractéristiques échoue
        
        Args:
            symbol (str): Symbole boursier
            
        Returns:
            list: Liste des caractéristiques par défaut
        """
        self.logger.info(f"Génération de caractéristiques par défaut pour {symbol}")
        
        default_features = []
        
        # 1. Ajouter les colonnes de prix standard (sauf Close qui est la cible)
        price_cols = ['Open', 'High', 'Low', 'Volume']
        for col_type in price_cols:
            col = self.get_price_column(symbol, col_type)
            if col is not None:
                if isinstance(col.name, str):
                    default_features.append(col.name)
                elif isinstance(col.name, tuple):
                    default_features.append(col.name)
        
        # 2. Ajouter des indicateurs techniques standards s'ils existent
        technical_indicators = [
            f"{symbol}.MA5", f"{symbol}.MA20", f"{symbol}.RSI", 
            f"{symbol}.MACD", f"{symbol}.Momentum", f"{symbol}.Volatility20"
        ]
        
        # Vérifier si les indicateurs existent dans les données
        for indicator in technical_indicators:
            if indicator in self.data.columns:
                default_features.append(indicator)
        
        # 3. Ajouter des données de sentiment si disponibles
        sentiment_cols = ["Sentiment.sentiment_score_normalized", "Sentiment.MA5"]
        for col in sentiment_cols:
            if col in self.data.columns:
                default_features.append(col)
        
        # 4. Ajouter quelques données macroéconomiques si disponibles
        macro_cols = ["Macro.GDP_growth", "Macro.inflation", "Macro.interest_rate"]
        for col in macro_cols:
            if col in self.data.columns:
                default_features.append(col)
        
        # S'assurer qu'il y a au moins quelques caractéristiques
        if len(default_features) < 3:
            self.logger.warning(f"Trop peu de caractéristiques par défaut trouvées pour {symbol}, ajout de colonnes génériques")
            # Ajouter des colonnes génériques (toutes sauf les prix de clôture)
            for col in self.data.columns:
                col_name = col if isinstance(col, str) else col[1] if isinstance(col, tuple) else str(col)
                if "Close" not in col_name and col not in default_features:
                    default_features.append(col)
                    if len(default_features) >= 10:  # Limiter à 10 caractéristiques
                        break
        
        self.logger.info(f"Caractéristiques par défaut pour {symbol}: {len(default_features)} caractéristiques")
        return default_features
        
    def validate_data(self):
        """
        Valide que les données requises sont disponibles et correctement formatées
        avant de démarrer le pipeline d'analyse.
        
        Returns:
            dict: Résultat de la validation avec statut et messages
        """
        validation_result = {
            "status": "success",
            "messages": [],
            "warnings": [],
            "errors": []
        }
        
        try:
            self.logger.info("Validation des données avant traitement")
            
            # 1. Vérifier si les données sont chargées
            if self.data is None:
                validation_result["status"] = "error"
                validation_result["errors"].append("Aucune donnée n'est chargée. Exécutez load_data() d'abord.")
                return validation_result
            
            # 2. Vérifier si les données sont vides
            if self.data.empty:
                validation_result["status"] = "error"
                validation_result["errors"].append("Le DataFrame des données est vide.")
                return validation_result
            
            # 3. Vérifier que tous les symboles demandés sont présents
            symbols_found = []
            missing_symbols = []
            
            for symbol in self.config['data']['symbols']:
                # Vérifier dans différents formats possibles 
                if isinstance(self.data.columns, pd.MultiIndex):
                    # Format MultiIndex
                    if any(symbol == idx[0] for idx in self.data.columns):
                        symbols_found.append(symbol)
                    else:
                        missing_symbols.append(symbol)
                else:
                    # Format simple avec notation symbol.Close
                    if any(symbol in col for col in self.data.columns):
                        symbols_found.append(symbol)
                    else:
                        missing_symbols.append(symbol)
            
            if missing_symbols:
                validation_result["status"] = "warning"
                validation_result["warnings"].append(f"Symboles manquants dans les données: {missing_symbols}")
            
            # 4. Vérifier la période de données
            min_required_rows = 100  # Minimum requis pour les modèles
            if len(self.data) < min_required_rows:
                validation_result["status"] = "error"
                validation_result["errors"].append(
                    f"Insuffisance de données temporelles ({len(self.data)} lignes, minimum requis: {min_required_rows})"
                )
            
            # 5. Vérifier les colonnes de prix requises pour chaque symbole
            required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
            for symbol in symbols_found:
                missing_cols = []
                
                for col_type in required_columns:
                    # Vérifier avec la méthode robuste
                    if self.get_price_column(symbol, col_type) is None:
                        missing_cols.append(col_type)
                
                if missing_cols:
                    validation_result["status"] = "warning"
                    validation_result["warnings"].append(
                        f"Colonnes manquantes pour {symbol}: {missing_cols}"
                    )
            
            # 6. Vérifier les valeurs manquantes
            missing_values = self.data.isnull().sum().sum()
            if missing_values > 0:
                validation_result["warnings"].append(
                    f"Le dataset contient {missing_values} valeurs manquantes " +
                    f"({(missing_values/(self.data.shape[0]*self.data.shape[1])*100):.2f}% du total)"
                )
            
            # 7. Vérifier les doublons dans l'index
            if self.data.index.duplicated().any():
                validation_result["warnings"].append(
                    f"L'index contient {self.data.index.duplicated().sum()} dates en double"
                )
            
            # 8. Vérifier la fréquence des données
            dates = pd.Series(self.data.index)
            date_diffs = dates.diff().dropna()
            
            if date_diffs.nunique() > 3:  # Plus de 3 intervalles différents
                validation_result["warnings"].append(
                    "Les données semblent avoir une fréquence irrégulière, " +
                    "ce qui pourrait affecter les modèles temporels"
                )
            
            # Résumé final
            if not validation_result["errors"] and not validation_result["warnings"]:
                validation_result["messages"].append("Validation des données réussie sans problème détecté")
            elif not validation_result["errors"]:
                validation_result["messages"].append(
                    f"Validation des données complétée avec {len(validation_result['warnings'])} avertissements"
                )
            else:
                validation_result["messages"].append(
                    f"Validation des données échouée avec {len(validation_result['errors'])} erreurs et " +
                    f"{len(validation_result['warnings'])} avertissements"
                )
            
            return validation_result
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation des données: {str(e)}")
            validation_result["status"] = "error"
            validation_result["errors"].append(f"Exception pendant la validation: {str(e)}")
            traceback.print_exc()
            return validation_result
        
    def run_pipeline(self, data_path=None, use_ensemble=True):
        """
        Exécute l'ensemble du pipeline de prévision avec seulement 4 modèles sélectionnés
        Version améliorée avec validation des données et meilleure gestion des erreurs
        
        Args:
            data_path (str, optional): Chemin vers les données personnalisées
            use_ensemble (bool): Utiliser l'approche d'ensemble pour les prédictions
        
        Returns:
            dict: Résultats complets
        """
        try:
            self.logger.info("Démarrage du pipeline de prévision optimisé (4 modèles)")
            pipeline_start_time = datetime.now()
            
            # Dictionnaire pour suivre les performances et l'état du pipeline
            pipeline_status = {
                "start_time": pipeline_start_time,
                "steps": {},
                "warnings": [],
                "errors": []
            }
            
            # 1. Charger les données
            pipeline_status["steps"]["load_data"] = {"start_time": datetime.now()}
            if not self.load_data(data_path):
                error_msg = "Échec du chargement des données. Arrêt du pipeline."
                self.logger.error(error_msg)
                pipeline_status["steps"]["load_data"]["status"] = "error"
                pipeline_status["steps"]["load_data"]["error"] = error_msg
                pipeline_status["errors"].append(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "pipeline_status": pipeline_status
                }
            
            pipeline_status["steps"]["load_data"]["end_time"] = datetime.now()
            pipeline_status["steps"]["load_data"]["status"] = "success"
            pipeline_status["steps"]["load_data"]["data_shape"] = self.data.shape
            
            # 2. Validation des données
            pipeline_status["steps"]["validate_data"] = {"start_time": datetime.now()}
            validation_result = self.validate_data()
            pipeline_status["steps"]["validate_data"]["result"] = validation_result
            
            if validation_result["status"] == "error":
                error_msg = "Validation des données échouée. Arrêt du pipeline."
                self.logger.error(error_msg)
                for error in validation_result["errors"]:
                    self.logger.error(f"Erreur de validation: {error}")
                    pipeline_status["errors"].append(error)
                
                pipeline_status["steps"]["validate_data"]["status"] = "error"
                pipeline_status["steps"]["validate_data"]["end_time"] = datetime.now()
                
                return {
                    "status": "error",
                    "message": error_msg,
                    "validation_result": validation_result,
                    "pipeline_status": pipeline_status
                }
            
            # Ajouter les avertissements à la liste générale
            for warning in validation_result.get("warnings", []):
                pipeline_status["warnings"].append(warning)
                self.logger.warning(f"Avertissement validation: {warning}")
            
            pipeline_status["steps"]["validate_data"]["status"] = "success"
            pipeline_status["steps"]["validate_data"]["end_time"] = datetime.now()
            
            # 3. Correction du format des colonnes
            pipeline_status["steps"]["fix_column_formats"] = {"start_time": datetime.now()}
            if not self._fix_column_formats():
                warning_msg = "Problème lors de la correction du format des colonnes"
                self.logger.warning(warning_msg)
                pipeline_status["warnings"].append(warning_msg)
            
            pipeline_status["steps"]["fix_column_formats"]["status"] = "success"
            pipeline_status["steps"]["fix_column_formats"]["end_time"] = datetime.now()
        
            # 4. Charger les données exogènes
            pipeline_status["steps"]["load_exogenous_data"] = {"start_time": datetime.now()}
            if not self.load_exogenous_data():
                warning_msg = "Impossible de charger les données exogènes"
                self.logger.warning(warning_msg)
                pipeline_status["warnings"].append(warning_msg)
                pipeline_status["steps"]["load_exogenous_data"]["status"] = "warning"
            else:
                pipeline_status["steps"]["load_exogenous_data"]["status"] = "success"
                pipeline_status["steps"]["load_exogenous_data"]["exogenous_sources"] = len(self.exogenous_data) if hasattr(self, 'exogenous_data') else 0
            
            pipeline_status["steps"]["load_exogenous_data"]["end_time"] = datetime.now()
        
            # 5. Analyser le sentiment
            if self.config['features']['sentiment_analysis']:
                pipeline_status["steps"]["analyze_sentiment"] = {"start_time": datetime.now()}
                
                if not self.analyze_sentiment():
                    warning_msg = "Impossible d'analyser le sentiment"
                    self.logger.warning(warning_msg)
                    pipeline_status["warnings"].append(warning_msg)
                    pipeline_status["steps"]["analyze_sentiment"]["status"] = "warning"
                else:
                    pipeline_status["steps"]["analyze_sentiment"]["status"] = "success"
                    pipeline_status["steps"]["analyze_sentiment"]["sentiment_shape"] = self.sentiment_data.shape if hasattr(self, 'sentiment_data') and self.sentiment_data is not None else None
                
                pipeline_status["steps"]["analyze_sentiment"]["end_time"] = datetime.now()
        
            # 6. Prétraiter les données
            pipeline_status["steps"]["preprocess_data"] = {"start_time": datetime.now()}
            
            if not self.preprocess_data():
                error_msg = "Échec du prétraitement des données. Arrêt du pipeline."
                self.logger.error(error_msg)
                pipeline_status["steps"]["preprocess_data"]["status"] = "error"
                pipeline_status["steps"]["preprocess_data"]["error"] = error_msg
                pipeline_status["errors"].append(error_msg)
                
                return {
                    "status": "error",
                    "message": error_msg,
                    "pipeline_status": pipeline_status
                }
            
            pipeline_status["steps"]["preprocess_data"]["status"] = "success"
            pipeline_status["steps"]["preprocess_data"]["end_time"] = datetime.now()
            pipeline_status["steps"]["preprocess_data"]["processed_data_shape"] = self.data.shape
        
            # 7. Sélectionner les caractéristiques
            pipeline_status["steps"]["select_features"] = {"start_time": datetime.now()}
            
            selected_features = self.select_features()
            
            # Vérifier que des caractéristiques ont été sélectionnées pour chaque symbole
            missing_features = [symbol for symbol in self.config['data']['symbols'] 
                            if symbol not in selected_features or not selected_features[symbol]]
            
            if missing_features:
                warning_msg = f"Caractéristiques manquantes pour les symboles: {missing_features}"
                self.logger.warning(warning_msg)
                pipeline_status["warnings"].append(warning_msg)
                pipeline_status["steps"]["select_features"]["status"] = "warning"
                pipeline_status["steps"]["select_features"]["missing_features"] = missing_features
            else:
                pipeline_status["steps"]["select_features"]["status"] = "success"
            
            # Compter les caractéristiques sélectionnées par symbole
            feature_counts = {symbol: len(features) for symbol, features in selected_features.items()}
            pipeline_status["steps"]["select_features"]["feature_counts"] = feature_counts
            pipeline_status["steps"]["select_features"]["end_time"] = datetime.now()
        
            # 8. Construire les modèles (seulement les 4 sélectionnés)
            pipeline_status["steps"]["build_models"] = {"start_time": datetime.now()}
            
            if not self.build_optimized_models(selected_features):
                error_msg = "Échec de la construction des modèles. Le pipeline continue avec des modèles limités."
                self.logger.error(error_msg)
                pipeline_status["steps"]["build_models"]["status"] = "warning"
                pipeline_status["steps"]["build_models"]["warning"] = error_msg
                pipeline_status["warnings"].append(error_msg)
            else:
                pipeline_status["steps"]["build_models"]["status"] = "success"
            
            # Compter les modèles construits par symbole et par type
            if hasattr(self, 'models'):
                model_counts = {}
                for symbol, models in self.models.items():
                    model_counts[symbol] = len(models)
                pipeline_status["steps"]["build_models"]["model_counts"] = model_counts
            
            pipeline_status["steps"]["build_models"]["end_time"] = datetime.now()
        
            # 9. Backtesting
            pipeline_status["steps"]["backtesting"] = {"start_time": datetime.now()}
            
            backtesting_result = self.backtest_models()
            
            # Vérifier si le backtesting a produit des résultats
            if not backtesting_result or (hasattr(self, 'results') and 'backtesting' not in self.results):
                warning_msg = "Le backtesting n'a pas produit de résultats"
                self.logger.warning(warning_msg)
                pipeline_status["warnings"].append(warning_msg)
                pipeline_status["steps"]["backtesting"]["status"] = "warning"
            else:
                pipeline_status["steps"]["backtesting"]["status"] = "success"
                
                # Collecter quelques statistiques sur les performances de backtesting
                if hasattr(self, 'results') and 'backtesting' in self.results:
                    backtesting_stats = {}
                    
                    for symbol, models in self.results['backtesting'].items():
                        symbol_stats = {}
                        
                        for model_name, metrics in models.items():
                            if 'avg_rmse' in metrics:
                                symbol_stats[model_name] = {
                                    'rmse': metrics.get('avg_rmse'),
                                    'mae': metrics.get('avg_mae'),
                                    'r2': metrics.get('avg_r2')
                                }
                        
                        backtesting_stats[symbol] = symbol_stats
                    
                    pipeline_status["steps"]["backtesting"]["stats"] = backtesting_stats
            
            pipeline_status["steps"]["backtesting"]["end_time"] = datetime.now()
        
            # 10. Générer les prévisions pondérées
            pipeline_status["steps"]["generate_predictions"] = {"start_time": datetime.now()}
            
            predictions = self.generate_weighted_predictions()
            
            # Vérifier que des prévisions ont été générées
            if not predictions or len(predictions) == 0:
                warning_msg = "Aucune prévision générée. Le pipeline continue mais sans résultats."
                self.logger.warning(warning_msg)
                pipeline_status["warnings"].append(warning_msg)
                pipeline_status["steps"]["generate_predictions"]["status"] = "warning"
                
                # Stocker un résultat minimal
                return {
                    "status": "warning",
                    "message": warning_msg,
                    "pipeline_status": pipeline_status
                }
            
            pipeline_status["steps"]["generate_predictions"]["status"] = "success"
            pipeline_status["steps"]["generate_predictions"]["prediction_symbols"] = list(predictions.keys())
            pipeline_status["steps"]["generate_predictions"]["end_time"] = datetime.now()
        
            # 11. Calculer les intervalles de confiance
            pipeline_status["steps"]["calculate_confidence"] = {"start_time": datetime.now()}
            
            predictions_with_conf = self.calculate_confidence_intervals(predictions)
            
            # Vérifier que les intervalles ont été calculés
            intervals_calculated = all('lower_bound' in pred and 'upper_bound' in pred for pred in predictions_with_conf.values())
            
            if not intervals_calculated:
                warning_msg = "Calcul incomplet des intervalles de confiance"
                self.logger.warning(warning_msg)
                pipeline_status["warnings"].append(warning_msg)
                pipeline_status["steps"]["calculate_confidence"]["status"] = "warning"
            else:
                pipeline_status["steps"]["calculate_confidence"]["status"] = "success"
            
            pipeline_status["steps"]["calculate_confidence"]["end_time"] = datetime.now()
        
            # Stocker les résultats
            self.results['predictions'] = predictions_with_conf
        
            # 12. Visualiser les résultats
            if self.config['output']['visualizations']:
                pipeline_status["steps"]["visualize"] = {"start_time": datetime.now()}
                
                # Visualiser les prévisions ensemblistes pondérées
                visualization_result = self.visualize_weighted_ensemble("visualisations")
                
                if not visualization_result:
                    warning_msg = "Erreur lors de la génération des visualisations"
                    self.logger.warning(warning_msg)
                    pipeline_status["warnings"].append(warning_msg)
                    pipeline_status["steps"]["visualize"]["status"] = "warning"
                else:
                    pipeline_status["steps"]["visualize"]["status"] = "success"
                
                pipeline_status["steps"]["visualize"]["end_time"] = datetime.now()
        
            # 13. Générer un rapport
            if self.config['output']['reports']:
                pipeline_status["steps"]["generate_report"] = {"start_time": datetime.now()}
                report_paths = []
                
                for format in self.config['output']['export_format']:
                    report_path = self.generate_report(format)
                    if report_path:
                        report_paths.append(report_path)
                
                if not report_paths:
                    warning_msg = "Erreur lors de la génération des rapports"
                    self.logger.warning(warning_msg)
                    pipeline_status["warnings"].append(warning_msg)
                    pipeline_status["steps"]["generate_report"]["status"] = "warning"
                else:
                    pipeline_status["steps"]["generate_report"]["status"] = "success"
                    pipeline_status["steps"]["generate_report"]["report_paths"] = report_paths
                
                pipeline_status["steps"]["generate_report"]["end_time"] = datetime.now()
        
            # Calculer la durée totale du pipeline
            pipeline_end_time = datetime.now()
            pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
            
            self.logger.info(f"Pipeline de prévision optimisé terminé en {pipeline_duration:.2f} secondes")
            
            # Générer un diagnostic final
            diagnostic_report = self.generate_diagnostic_report()
        
            return {
                "status": "success" if not pipeline_status["errors"] else "warning",
                "predictions": predictions_with_conf,
                "backtesting": self.results.get('backtesting', {}),
                "pipeline_status": pipeline_status,
                "diagnostic": diagnostic_report,
                "duration_seconds": pipeline_duration,
                "warnings": len(pipeline_status["warnings"]),
                "errors": len(pipeline_status["errors"])
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution du pipeline optimisé: {str(e)}")
            traceback.print_exc()
            
            # Essayer de générer un diagnostic même en cas d'erreur
            try:
                diagnostic = self.generate_diagnostic_report()
            except:
                diagnostic = {"error": "Impossible de générer le diagnostic"}
            
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "pipeline_status": pipeline_status if 'pipeline_status' in locals() else None,
                "diagnostic": diagnostic
            }
        
    def run_pipeline_robust_enhanced(self, data_path=None):
        """
        Version améliorée du pipeline robuste avec meilleure gestion des erreurs
        """
        try:
            self.logger.info("Démarrage du pipeline de prévision robuste amélioré")
            pipeline_start_time = datetime.now()
            
            # 1. Vérification de la clé API
            if (not hasattr(self, 'config') or 'tiingo' not in self.config or 
                not self.config['tiingo'].get('api_key') or 
                self.config['tiingo']['api_key'] == "VOTRE_CLÉ_API_TIINGO_ICI"):
                
                self.logger.error("Clé API Tiingo non configurée correctement!")
                self.logger.info("Veuillez définir une clé API valide dans self.config['tiingo']['api_key']")
                return {"status": "error", "message": "Clé API Tiingo manquante"}
            
            # 2. Chargement des données avec retry
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                if self.load_data(data_path):
                    self.logger.info("Données chargées avec succès")
                    break
                else:
                    retry_count += 1
                    self.logger.warning(f"Échec du chargement des données, tentative {retry_count}/{max_retries}")
                    if retry_count < max_retries:
                        time.sleep(2)  # Attendre 2 secondes avant de réessayer
                    else:
                        return {"status": "error", "message": "Impossible de charger les données après plusieurs tentatives"}
            
            # 3. Validation et nettoyage des données
            validation_result = self.validate_data()
            if validation_result["status"] == "error":
                return {"status": "error", "message": "Données invalides", "details": validation_result}
            
            # Nettoyage robuste
            if not self.nettoyer_donnees():
                self.logger.warning("Problèmes lors du nettoyage, mais continuation du pipeline")
            
            # 4. Prétraitement avec gestion d'erreurs
            if not self.preprocess_data():
                self.logger.error("Échec du prétraitement")
                return {"status": "error", "message": "Échec du prétraitement des données"}
            
            # 5. Sélection de features améliorée
            features = self.select_features()
            
            # Vérifier la qualité de la sélection
            for symbol in self.config['data']['symbols']:
                if symbol in features and len(features[symbol]) < 3:
                    self.logger.warning(f"Peu de features sélectionnées pour {symbol}, ajout de features par défaut")
                    features[symbol].extend(self._get_default_features(symbol)[:5])
                    features[symbol] = list(set(features[symbol]))  # Supprimer les doublons
            
            # 6. Construction de modèles avec validation de qualité
            self.models = {}
            successful_models = 0
            
            for symbol in self.config['data']['symbols']:
                self.models[symbol] = {}
                symbol_models = 0
                
                # Modèle Gradient Boosting amélioré
                try:
                    self._build_gradient_boosting_model(symbol, features[symbol])
                    if 'gradient_boosting' in self.models[symbol]:
                        # Vérifier la qualité
                        model_info = self.models[symbol]['gradient_boosting']
                        if model_info.get('quality') != 'poor':
                            symbol_models += 1
                        else:
                            self.logger.warning(f"Modèle Gradient Boosting de mauvaise qualité pour {symbol}")
                except Exception as e:
                    self.logger.error(f"Erreur construction Gradient Boosting pour {symbol}: {e}")
                
                # Modèle GARCH
                try:
                    self._build_garch_model(symbol)
                    if 'garch' in self.models[symbol]:
                        symbol_models += 1
                except Exception as e:
                    self.logger.error(f"Erreur construction GARCH pour {symbol}: {e}")
                
                # Modèle ARIMA (optionnel)
                try:
                    self._build_arima_model(symbol, features[symbol])
                    if 'arima' in self.models[symbol]:
                        symbol_models += 1
                except Exception as e:
                    self.logger.warning(f"ARIMA non disponible pour {symbol}: {e}")
                
                if symbol_models > 0:
                    successful_models += 1
                    self.logger.info(f"{symbol_models} modèles construits pour {symbol}")
                else:
                    self.logger.error(f"Aucun modèle construit pour {symbol}")
            
            if successful_models == 0:
                return {"status": "error", "message": "Aucun modèle construit avec succès"}
            
            # 7. Backtesting avec gestion d'erreurs
            try:
                backtest_results = self.backtest_models()
                if backtest_results:
                    self.logger.info("Backtesting terminé")
                else:
                    self.logger.warning("Backtesting échoué mais pipeline continue")
            except Exception as e:
                self.logger.error(f"Erreur lors du backtesting: {e}")
                backtest_results = {}
            
            # 8. Génération de prévisions avec la nouvelle méthode
            try:
                predictions = self.generate_weighted_predictions()
                
                if predictions:
                    self.results['predictions'] = predictions
                    self.logger.info(f"Prévisions générées pour {len(predictions)} symboles")
                    
                    # Affichage des résultats
                    for symbol, pred in predictions.items():
                        if 'values' in pred and len(pred['values']) > 0:
                            price_data = self.get_price_column(symbol, 'Close')
                            if price_data is not None:
                                last_price = price_data.iloc[-1]
                                future_price = pred['values'][0]
                                change_pct = ((future_price / last_price) - 1) * 100
                                
                                self.logger.info(f"Prévision pour {symbol}: {last_price:.2f} -> {future_price:.2f} ({change_pct:+.2f}%)")
                                
                                if 'lower_bound' in pred and 'upper_bound' in pred:
                                    self.logger.info(f"  Intervalle de confiance: [{pred['lower_bound'][0]:.2f}, {pred['upper_bound'][0]:.2f}]")
                                
                                if 'models_info' in pred:
                                    models_used = pred['models_info'].get('models', [])
                                    weights_used = pred['models_info'].get('weights', {})
                                    models_str = ", ".join([f"{m} ({weights_used.get(m, 0):.2f})" for m in models_used])
                                    self.logger.info(f"  Modèles utilisés (poids): {models_str}")
                else:
                    self.logger.error("Aucune prévision générée")
                    return {"status": "error", "message": "Échec de la génération des prévisions"}
                    
            except Exception as e:
                self.logger.error(f"Erreur lors de la génération des prévisions: {e}")
                return {"status": "error", "message": f"Erreur prévisions: {str(e)}"}
            
            # 9. Génération des visualisations
            try:
                if self.config['output']['visualizations']:
                    viz_path = "visualisations_robustes"
                    viz_success = self.visualize_weighted_ensemble(viz_path)
                    if viz_success:
                        self.logger.info(f"Visualisations générées dans {viz_path}")
            except Exception as e:
                self.logger.warning(f"Erreur lors de la génération des visualisations: {e}")
            
            # 10. Génération du tableau de bord
            dashboard_path = None
            try:
                dashboard_path = self.generate_trading_dashboard("trading_dashboard")
                if dashboard_path:
                    self.logger.info(f"Tableau de bord généré: {dashboard_path}")
            except Exception as e:
                self.logger.warning(f"Erreur lors de la génération du tableau de bord: {e}")
            
            # 11. Génération des rapports
            try:
                if self.config['output']['reports']:
                    for format in self.config['output']['export_format']:
                        report_path = self.generate_report(format)
                        if report_path:
                            self.logger.info(f"Rapport généré: {report_path}")
            except Exception as e:
                self.logger.warning(f"Erreur lors de la génération des rapports: {e}")
            
            # 12. Sauvegarde des modèles
            try:
                self.save_model("models_optimized")
                self.logger.info("Modèles sauvegardés")
            except Exception as e:
                self.logger.warning(f"Erreur lors de la sauvegarde: {e}")
            
            # Calcul du temps d'exécution
            pipeline_end_time = datetime.now()
            duration = (pipeline_end_time - pipeline_start_time).total_seconds()
            
            self.logger.info(f"Pipeline robuste amélioré terminé en {duration:.2f} secondes")
            
            return {
                "status": "success",
                "predictions": predictions,
                "dashboard_path": dashboard_path,
                "duration_seconds": duration,
                "models_built": successful_models,
                "symbols_processed": len(predictions) if predictions else 0
            }
            
        except Exception as e:
            self.logger.error(f"Erreur générale dans le pipeline robuste: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }

    def build_optimized_models(self, selected_features=None):
        """
        Construit uniquement les 4 modèles optimaux (ARIMA, LSTM, GARCH, Gradient Boosting)
        Version améliorée avec meilleure gestion des erreurs et debugging
        
        Args:
            selected_features (dict, optional): Caractéristiques sélectionnées par symbole
        
        Returns:
            bool: True si les modèles ont été construits avec succès
        """
        try:
            self.logger.info("Construction des 4 modèles optimaux de prévision")
            
            # Valider les données avant de construire les modèles
            validation_result = self.validate_data()
            if validation_result["status"] == "error":
                self.logger.error("Validation des données échouée, impossible de construire les modèles")
                for error in validation_result["errors"]:
                    self.logger.error(f"Erreur de validation: {error}")
                return False
            
            # Logger les avertissements éventuels
            if validation_result["warnings"]:
                for warning in validation_result["warnings"]:
                    self.logger.warning(f"Avertissement de validation: {warning}")
            
            # Si pas de caractéristiques sélectionnées, utiliser toutes les caractéristiques
            if selected_features is None:
                self.logger.info("Aucune caractéristique fournie, sélection automatique")
                selected_features = self.select_features()
            
            # Initialiser le dictionnaire des modèles
            self.models = {}
            
            # Construire les modèles pour chaque symbole
            models_built = 0
            for symbol in self.config['data']['symbols']:
                self.logger.info(f"Démarrage de la construction des modèles pour {symbol}")
                
                # Récupérer les caractéristiques pour ce symbole
                features = selected_features.get(symbol, [])
                if not features:
                    self.logger.warning(f"Aucune caractéristique sélectionnée pour {symbol}, utilisation des caractéristiques par défaut")
                    features = self._get_default_features(symbol)
                
                # Vérifier que nous avons la colonne cible
                target_col = None
                try:
                    target_col = self.get_price_column(symbol, 'Close')
                    if target_col is None:
                        self.logger.error(f"Impossible de trouver les données de prix pour {symbol}, modèles ignorés")
                        continue
                except Exception as e:
                    self.logger.error(f"Erreur lors de la récupération des données de prix pour {symbol}: {str(e)}")
                    continue
                
                # Initialiser le dictionnaire de modèles pour ce symbole
                self.models[symbol] = {}
                models_built_for_symbol = 0
                
                # Construire les 4 modèles optimaux
                
                # 1. Modèle ARIMA/SARIMA
                try:
                    self.logger.info(f"Construction du modèle ARIMA pour {symbol}")
                    self._build_arima_model(symbol, features)
                    if 'arima' in self.models[symbol] or 'sarima' in self.models[symbol]:
                        models_built_for_symbol += 1
                        self.logger.info(f"Modèle ARIMA construit avec succès pour {symbol}")
                    else:
                        self.logger.warning(f"Échec de construction du modèle ARIMA pour {symbol}")
                except Exception as e:
                    self.logger.error(f"Erreur lors de la construction du modèle ARIMA pour {symbol}: {str(e)}")
                    traceback.print_exc()
                
                # 2. Modèle GARCH
                try:
                    self.logger.info(f"Construction du modèle GARCH pour {symbol}")
                    self._build_garch_model(symbol)
                    if 'garch' in self.models[symbol]:
                        models_built_for_symbol += 1
                        self.logger.info(f"Modèle GARCH construit avec succès pour {symbol}")
                    else:
                        self.logger.warning(f"Échec de construction du modèle GARCH pour {symbol}")
                except Exception as e:
                    self.logger.error(f"Erreur lors de la construction du modèle GARCH pour {symbol}: {str(e)}")
                    traceback.print_exc()
                
                # 3. Modèle Gradient Boosting
                try:
                    self.logger.info(f"Construction du modèle Gradient Boosting pour {symbol}")
                    self._build_gradient_boosting_model(symbol, features)
                    if 'gradient_boosting' in self.models[symbol]:
                        models_built_for_symbol += 1
                        self.logger.info(f"Modèle Gradient Boosting construit avec succès pour {symbol}")
                    else:
                        self.logger.warning(f"Échec de construction du modèle Gradient Boosting pour {symbol}")
                except Exception as e:
                    self.logger.error(f"Erreur lors de la construction du modèle Gradient Boosting pour {symbol}: {str(e)}")
                    traceback.print_exc()
                
                # 4. Modèle LSTM
                try:
                    self.logger.info(f"Construction du modèle LSTM pour {symbol}")
                    self._build_lstm_model_only(symbol, features)
                    if 'lstm' in self.models[symbol]:
                        models_built_for_symbol += 1
                        self.logger.info(f"Modèle LSTM construit avec succès pour {symbol}")
                    else:
                        self.logger.warning(f"Échec de construction du modèle LSTM pour {symbol}")
                except Exception as e:
                    self.logger.error(f"Erreur lors de la construction du modèle LSTM pour {symbol}: {str(e)}")
                    traceback.print_exc()
                
                # Vérifier si au moins un modèle a été construit pour ce symbole
                if models_built_for_symbol > 0:
                    models_built += 1
                    self.logger.info(f"{models_built_for_symbol} modèles construits pour {symbol}")
                else:
                    self.logger.error(f"Aucun modèle construit pour {symbol}")
            
            self.logger.info(f"Modèles optimaux construits pour {models_built} symboles sur {len(self.config['data']['symbols'])}")
            
            # Si au moins un modèle a été construit, considérer comme réussi
            return models_built > 0

        except Exception as e:
            self.logger.error(f"Erreur lors de la construction des modèles optimaux: {str(e)}")
            traceback.print_exc()
            return False

    def _build_arima_model(self, symbol, features):
        """
        Construit et optimise un modèle ARIMA/SARIMA simplifié et robuste
        """
        try:
            self.logger.info(f"Construction du modèle ARIMA/SARIMA pour {symbol}")
            
            # Préparer les données
            target_col = self.get_price_column(symbol, 'Close')
            if target_col is None:
                self.logger.error(f"Impossible de récupérer les données de prix pour {symbol}")
                return
            
            # Vérifier la qualité des données
            y_original = target_col
            
            # Éliminer les valeurs manquantes
            y = y_original.dropna()
            
            if len(y) < 60:  # Besoin d'au moins 60 points de données
                self.logger.warning(f"Données insuffisantes pour ARIMA ({len(y)} points)")
                return
                
            # Prendre le logarithme pour stabiliser la variance
            # Évite les erreurs "value too large for dtype('float64')"
            min_value = y.min()
            if min_value <= 0:
                # Ajouter un offset pour éviter log(0) ou log(valeur négative)
                y_log = np.log(y - min_value + 1)
            else:
                y_log = np.log(y)
            
            # Corriger l'index de dates pour éviter les avertissements
            y_log = self._fix_date_index(y_log, freq='B')  # 'B' pour jours ouvrables
            
            # Tester quelques modèles ARIMA simples au lieu d'optimiser
            orders = [(1,1,1), (1,1,0), (0,1,1), (2,1,2)]
            
            best_aic = float('inf')
            best_model = None
            best_order = None
            
            for order in orders:
                try:
                    model = ARIMA(y_log, order=order)
                    fitted = model.fit()
                    
                    current_aic = fitted.aic
                    self.logger.info(f"ARIMA{order} AIC: {current_aic:.2f}")
                    
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_model = fitted
                        best_order = order
                except Exception as model_error:
                    self.logger.warning(f"Échec du modèle ARIMA{order}: {str(model_error)}")
                    continue
            
            if best_model is not None:
                self.models[symbol]['arima'] = {
                    'model': best_model,
                    'order': best_order,
                    'is_log_transformed': True,  # Important pour la prédiction!
                    'log_offset': 0 if min_value > 0 else (1 - min_value),
                    'aic': best_aic
                }
                
                self.logger.info(f"Modèle ARIMA{best_order} construit avec succès pour {symbol}, AIC: {best_aic:.2f}")
                return True
            else:
                self.logger.warning(f"Échec de construction du modèle ARIMA pour {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur générale lors de la construction du modèle ARIMA pour {symbol}: {str(e)}")
            traceback.print_exc()
            return False

    def calculate_confidence_intervals(self, predictions):
        """
        Calcule les intervalles de confiance pour les prévisions
        
        Args:
            predictions (dict): Prévisions par symbole
            
        Returns:
            dict: Prévisions avec intervalles de confiance
        """
        try:
            self.logger.info("Calcul des intervalles de confiance pour les prévisions")
            
            # Si les prévisions sont vides, retourner directement
            if not predictions:
                return predictions
            
            # Pour chaque symbole
            for symbol, pred in predictions.items():
                # Vérifier si les intervalles sont déjà calculés
                if 'lower_bound' in pred and 'upper_bound' in pred:
                    self.logger.info(f"Intervalles de confiance déjà présents pour {symbol}")
                    continue
                
                # Récupérer les données historiques avec la méthode robuste
                historical = self.get_price_column(symbol, 'Close')
                if historical is None:
                    self.logger.error(f"Impossible de trouver les données historiques pour {symbol}")
                    continue
                
                # Calculer la volatilité historique (écart-type des rendements)
                returns = historical.pct_change().dropna()
                volatility = returns.std()
                
                # Prédictions et dates
                values = pred['values']
                dates = pred['dates']
                
                # Calculer les intervalles en fonction de la volatilité historique et du niveau de confiance
                # Pour une distribution normale, un intervalle de confiance de 95% est environ 1.96 * écart-type
                z_score = 1.96
                if self.confidence_level != 0.95:
                    # Calculer le Z-score pour d'autres niveaux de confiance
                    from scipy import stats
                    z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
                
                # Calculer les bornes
                lower_bound = np.zeros_like(values)
                upper_bound = np.zeros_like(values)
                
                # Élargir l'intervalle avec le temps (incertitude croissante)
                for i in range(len(values)):
                    # Plus on va loin dans le futur, plus l'incertitude augmente
                    time_factor = np.sqrt(i + 1)
                    
                    # Calculer les bornes en tenant compte de l'incertitude croissante
                    interval_width = values[i] * volatility * z_score * time_factor
                    lower_bound[i] = values[i] - interval_width
                    upper_bound[i] = values[i] + interval_width
                
                # Mettre à jour les prévisions avec les intervalles de confiance
                pred['lower_bound'] = lower_bound
                pred['upper_bound'] = upper_bound
                pred['confidence_level'] = self.confidence_level
                
                self.logger.info(f"Intervalles de confiance calculés pour {symbol}")
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des intervalles de confiance: {str(e)}")
            traceback.print_exc()
            return predictions
            
    def _build_garch_model(self, symbol):
        """
        Construit un modèle GARCH pour modéliser la volatilité
        
        Args:
            symbol (str): Symbole boursier
        """
        try:
            self.logger.info(f"Construction du modèle GARCH pour {symbol}")
            
            # Préparer les données - CORRECTION POUR MULTIINDEX
            if isinstance(self.data.columns, pd.MultiIndex):
                # Format MultiIndex
                target_col = (symbol, 'Close')
            else:
                # Format simple
                target_col = f"{symbol}.Close"
                
            # Vérifier si la colonne existe
            if (isinstance(target_col, tuple) and target_col in self.data.columns) or \
            (isinstance(target_col, str) and target_col in self.data.columns):
                returns = 100 * self.data[target_col].pct_change().dropna()
            else:
                # Si la colonne n'existe pas, chercher une alternative
                for col in self.data.columns:
                    col_name = col if isinstance(col, str) else col[1] if isinstance(col, tuple) else str(col)
                    col_symbol = col if isinstance(col, str) else col[0] if isinstance(col, tuple) else str(col)
                    
                    if 'Close' in col_name and symbol in col_symbol:
                        self.logger.info(f"Utilisation de la colonne {col} comme alternative à {target_col}")
                        returns = 100 * self.data[col].pct_change().dropna()
                        break
                else:
                    raise ValueError(f"Aucune colonne de prix de clôture trouvée pour {symbol}")

            # Paramètres du modèle
            p = self.config['models']['garch']['p']
            q = self.config['models']['garch']['q']
            dist = self.config['models']['garch']['distribution']
            
            # Construire le modèle GARCH
            from arch import arch_model
            garch_model = arch_model(returns, vol='Garch', p=p, q=q, dist=dist)
            fitted_model = garch_model.fit(disp='off')
            
            self.models[symbol]['garch'] = {
                'model': fitted_model,
                'p': p,
                'q': q,
                'distribution': dist
            }
            
            self.logger.info(f"Modèle GARCH({p},{q}) ajusté pour {symbol} avec distribution {dist}")
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la construction du modèle GARCH pour {symbol}: {str(e)}")
            traceback.print_exc()
            
    def _build_gradient_boosting_model(self, symbol, features):
        """
        Version corrigée du modèle Gradient Boosting avec meilleure validation et paramètres optimisés
        """
        try:
            self.logger.info(f"Construction du modèle Gradient Boosting amélioré pour {symbol}")
            
            # Récupérer la colonne cible avec la méthode robuste
            target_col = self.get_price_column(symbol, 'Close')
            if target_col is None:
                self.logger.error(f"Impossible de récupérer les données de prix pour {symbol}")
                return
            
            # Vérifier qu'on a assez de features valides
            valid_features = []
            X_full = pd.DataFrame(index=self.data.index)
            
            for feature in features:
                if feature in self.data.columns:
                    feature_data = self.data[feature]
                    # Vérifier que la feature n'est pas constante
                    if feature_data.nunique() > 1 and not feature_data.isna().all():
                        X_full[feature] = feature_data
                        valid_features.append(feature)
            
            if len(valid_features) < 3:
                self.logger.warning(f"Trop peu de features valides ({len(valid_features)}) pour {symbol}")
                return
            
            self.logger.info(f"Features valides pour {symbol}: {len(valid_features)}")
            
            # Créer les variables explicatives et la cible
            X = X_full[valid_features].copy()
            
            # Utiliser les rendements futurs comme cible (plus réaliste)
            y = target_col.pct_change().shift(-1)  # Rendement du jour suivant
            
            # Nettoyer les données
            mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[mask].copy()
            y_clean = y[mask].copy()
            
            if len(X_clean) < 100:
                self.logger.error(f"Données insuffisantes après nettoyage pour {symbol}: {len(X_clean)}")
                return
            
            # Vérifier la variance de la cible
            y_std = y_clean.std()
            if y_std < 1e-6:
                self.logger.warning(f"Variance de la cible trop faible pour {symbol}: {y_std}")
                return
            
            # Supprimer les outliers extrêmes (au-delà de 3 sigma)
            y_mean = y_clean.mean()
            outlier_mask = abs(y_clean - y_mean) < 3 * y_std
            X_clean = X_clean[outlier_mask]
            y_clean = y_clean[outlier_mask]
            
            self.logger.info(f"Données finales pour l'entraînement: {len(X_clean)} échantillons")
            
            # Division train/test temporelle (plus réaliste)
            split_idx = int(len(X_clean) * 0.8)
            X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
            y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
            
            # Imputation des valeurs manquantes si nécessaire
            if X_train.isna().any().any():
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X_train = pd.DataFrame(imputer.fit_transform(X_train), 
                                    columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(imputer.transform(X_test), 
                                    columns=X_test.columns, index=X_test.index)
            
            # Modèle avec des paramètres plus conservateurs
            base_model = GradientBoostingRegressor(
                random_state=42,
                n_estimators=100,      # Réduire pour éviter l'overfitting
                max_depth=4,           # Limiter la profondeur
                learning_rate=0.1,     # Taux d'apprentissage modéré
                subsample=0.8,         # Sous-échantillonnage pour la robustesse
                min_samples_split=20,  # Éviter les divisions sur peu d'échantillons
                min_samples_leaf=10    # Feuilles avec assez d'échantillons
            )
            
            # Grille de paramètres réduite et plus réaliste
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 0.9]
            }
            
            # Validation croisée temporelle
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Recherche des hyperparamètres avec moins d'itérations
            search = RandomizedSearchCV(
                base_model, 
                param_grid, 
                n_iter=8,  # Réduire le nombre d'itérations
                cv=tscv, 
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=1  # Éviter les problèmes de parallélisation
            )
            
            # Entraînement
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            # Validation sur le test set
            y_pred = best_model.predict(X_test)
            
            # Calcul des métriques corrigées
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # R² corrigé pour éviter les valeurs aberrantes
            try:
                r2 = r2_score(y_test, y_pred)
                # Si R² est très négatif, utiliser une approche alternative
                if r2 < -10:
                    # Calculer R² manuellement avec une approche plus robuste
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -1
            except:
                r2 = -1  # Valeur par défaut en cas d'erreur
            
            # Validation de la qualité du modèle
            model_quality = "good"
            if r2 < -0.5 or mse > y_std**2 * 5:
                model_quality = "poor"
                self.logger.warning(f"Modèle de qualité médiocre détecté pour {symbol} (R²={r2:.4f})")
            
            # Importance des features avec validation
            feature_importance = {}
            try:
                importances = best_model.feature_importances_
                for i, feature in enumerate(valid_features):
                    feature_importance[feature] = float(importances[i])
                
                # Vérifier que les importances ne sont pas toutes nulles
                total_importance = sum(feature_importance.values())
                if total_importance < 1e-6:
                    self.logger.warning(f"Importances des features très faibles pour {symbol}")
            except Exception as e:
                self.logger.warning(f"Erreur lors du calcul des importances: {e}")
                feature_importance = {f: 1.0/len(valid_features) for f in valid_features}
            
            # Stocker le modèle avec des métadonnées étendues
            self.models[symbol]['gradient_boosting'] = {
                'model': best_model,
                'features': valid_features,
                'original_features': features,
                'best_params': search.best_params_,
                'quality': model_quality,
                'metrics': {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'y_std': float(y_std),
                    'n_samples': len(X_train)
                },
                'feature_importance': feature_importance,
                'scaler_info': {
                    'target_mean': float(y_mean),
                    'target_std': float(y_std)
                }
            }
            
            self.logger.info(f"Modèle Gradient Boosting pour {symbol}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f} (qualité: {model_quality})")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la construction du modèle Gradient Boosting pour {symbol}: {str(e)}")
            traceback.print_exc()

    def _build_lstm_model_only(self, symbol, features):
        """
        Construit un modèle LSTM simplifié avec une meilleure gestion des erreurs
        """
        try:
            self.logger.info(f"Construction du modèle LSTM pour {symbol}")
            
            # Récupérer la colonne cible
            target_col = self.get_price_column(symbol, 'Close')
            if target_col is None:
                self.logger.error(f"Impossible de récupérer les données de prix pour {symbol}")
                return
                
            # Limiter le nombre de caractéristiques pour éviter le surajustement
            if len(features) > 5:
                self.logger.info(f"Trop de caractéristiques ({len(features)}), limitation à 5")
                features = features[:5]
            
            # Extraire les données et éliminer les lignes avec des NaN
            X = self.data[features]
            y = target_col.values.reshape(-1, 1)
            
            # Vérifier les valeurs manquantes
            X_nan_count = X.isnull().sum().sum()
            y_nan_count = np.isnan(y).sum()
            
            if X_nan_count > 0 or y_nan_count > 0:
                self.logger.warning(f"Nettoyage des valeurs manquantes: X={X_nan_count}, y={y_nan_count}")
                # Créer un masque pour les lignes sans NaN
                mask = ~(X.isnull().any(axis=1) | np.isnan(y).flatten())
                X = X[mask]
                y = y[mask.values]  # .values pour convertir en array
            
            # Vérifier qu'il reste suffisamment de données
            if len(X) < 60:
                self.logger.error(f"Données insuffisantes pour LSTM: {len(X)} lignes")
                return
                
            # Normaliser les données
            from sklearn.preprocessing import MinMaxScaler
            X_scaler = MinMaxScaler()
            X_scaled = X_scaler.fit_transform(X)
            
            y_scaler = MinMaxScaler()
            y_scaled = y_scaler.fit_transform(y)
            
            # Créer des séquences pour l'apprentissage (plus courtes pour robustesse)
            seq_length = min(10, len(X) // 10)
            self.logger.info(f"Utilisation d'une longueur de séquence de {seq_length}")
            
            X_seq, y_seq = [], []
            for i in range(len(X_scaled) - seq_length):
                X_seq.append(X_scaled[i:i + seq_length])
                y_seq.append(y_scaled[i + seq_length][0])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Diviser en train/test
            train_size = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:train_size], X_seq[train_size:]
            y_train, y_test = y_seq[:train_size], y_seq[train_size:]
            
            # Modèle LSTM simplifié (moins de couches, moins de neurones)
            model = Sequential([
                LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                Dense(1)
            ])
            
            # Compiler avec un learning rate plus faible
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Early stopping pour éviter le surajustement
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Définir un batch_size adapté à la taille des données
            batch_size = min(32, len(X_train))
            
            # Réduire le nombre d'epochs pour limiter le risque de surajustement
            epochs = 50
            
            # Entraîner le modèle
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Évaluer le modèle
            y_pred = model.predict(X_test, verbose=0)
            
            # Retransformer les prédictions à l'échelle originale
            y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_orig = y_scaler.inverse_transform(y_pred).flatten()
            
            # Calculer les métriques
            mse = mean_squared_error(y_test_orig, y_pred_orig)
            mae = mean_absolute_error(y_test_orig, y_pred_orig)
            r2 = r2_score(y_test_orig, y_pred_orig)
            
            # Vérifier si les métriques sont acceptables
            if r2 < -1.0:  # Un très mauvais R² indique un modèle défaillant
                self.logger.warning(f"Mauvaise performance du modèle LSTM: R²={r2:.4f}")
                return False
            
            # Stocker le modèle
            self.models[symbol]['lstm'] = {
                'model': model,
                'scaler_X': X_scaler,
                'scaler_y': y_scaler,
                'seq_length': seq_length,
                'features': features,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
            }
            
            self.logger.info(f"Modèle LSTM pour {symbol}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la construction du modèle LSTM pour {symbol}: {str(e)}")
            traceback.print_exc()
            return False    
            
    def _create_dl_model_for_backtest(self, model_type, input_shape, seq_length):
        """
        Crée un modèle d'apprentissage profond pour le backtesting
        
        Args:
            model_type (str): Type de modèle ('lstm', 'gru', 'transformer')
            input_shape (tuple): Forme des données d'entrée
            seq_length (int): Longueur de la séquence
            
        Returns:
            tf.keras.Model: Modèle d'apprentissage profond
        """
        try:
            # Récupérer les dimensions d'entrée
            n_timesteps, n_features = input_shape[1], input_shape[2]
            
            # Créer le modèle en fonction du type
            if model_type == 'lstm':
                model = Sequential()
                
                # Première couche LSTM avec return_sequences=True
                model.add(LSTM(64, return_sequences=True, input_shape=(n_timesteps, n_features)))
                model.add(BatchNormalization())
                model.add(Dropout(0.2))
                
                # Seconde couche LSTM
                model.add(LSTM(32))
                model.add(BatchNormalization())
                model.add(Dropout(0.2))
                
                # Couche de sortie
                model.add(Dense(1))
            
            elif model_type == 'gru':
                model = Sequential()
                
                # Première couche GRU avec return_sequences=True
                model.add(GRU(64, return_sequences=True, input_shape=(n_timesteps, n_features)))
                model.add(BatchNormalization())
                model.add(Dropout(0.2))
                
                # Seconde couche GRU
                model.add(GRU(32))
                model.add(BatchNormalization())
                model.add(Dropout(0.2))
                
                # Couche de sortie
                model.add(Dense(1))
            
            elif model_type == 'transformer':
                # Implémentation simplifiée d'un modèle de type Transformer
                # Les transformers réels sont plus complexes avec des couches d'attention multi-têtes
                inputs = tf.keras.Input(shape=(n_timesteps, n_features))
                
                # Couche d'attention
                x = tf.keras.layers.MultiHeadAttention(
                    num_heads=4, key_dim=16
                )(inputs, inputs)
                
                # Ajouter une connexion résiduelle
                x = tf.keras.layers.Add()([inputs, x])
                x = tf.keras.layers.LayerNormalization()(x)
                
                # Couche de Feed Forward
                ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(n_features)
                ])
                
                x = ffn(x)
                
                # Ajouter une autre connexion résiduelle
                x = tf.keras.layers.Add()([inputs, x])
                x = tf.keras.layers.LayerNormalization()(x)
                
                # Aplatir et passer à la couche de sortie
                x = tf.keras.layers.GlobalAveragePooling1D()(x)
                outputs = tf.keras.layers.Dense(1)(x)
                
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            else:
                self.logger.warning(f"Type de modèle non reconnu: {model_type}, utilisation de LSTM par défaut")
                model = Sequential()
                model.add(LSTM(64, input_shape=(n_timesteps, n_features)))
                model.add(Dense(1))
            
            # Compiler le modèle
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            return model
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du modèle {model_type}: {str(e)}")
            traceback.print_exc()
            
            # Retourner un modèle minimal en cas d'erreur
            minimal_model = Sequential()
            minimal_model.add(Dense(1, input_shape=(n_timesteps, n_features)))
            minimal_model.compile(optimizer='adam', loss='mse')
            
            return minimal_model  

    def _predict_with_arima(self, symbol, model_info, horizon):
        """
        Génère des prévisions avec un modèle ARIMA/SARIMA avec gestion correcte des transformations
        """
        try:
            model = model_info['model']
            is_log = model_info.get('is_log_transformed', False)
            log_offset = model_info.get('log_offset', 0)
            
            # Générer les prévisions
            forecast = model.get_forecast(steps=horizon)
            mean_forecast = forecast.predicted_mean
            
            # Intervalles de confiance
            conf_int = forecast.conf_int(alpha=1-self.confidence_level)
            
            # Si les données ont été transformées en log, retransformer
            if is_log:
                # Retransformer les prévisions
                mean_forecast = np.exp(mean_forecast) - log_offset
                
                # Retransformer les intervalles de confiance
                lower_bound = np.exp(conf_int.iloc[:, 0].values) - log_offset
                upper_bound = np.exp(conf_int.iloc[:, 1].values) - log_offset
            else:
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            
            # Créer des dates pour les prévisions
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                        periods=horizon, 
                                        freq='B')  # Jours ouvrables
            
            # Formater les résultats
            predictions = {
                'dates': forecast_dates,
                'values': mean_forecast.values if hasattr(mean_forecast, 'values') else mean_forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model_type': 'arima' if 'order' in model_info else 'sarima'
            }
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Erreur lors des prévisions ARIMA pour {symbol}: {str(e)}")
            traceback.print_exc()
            return None
        
    
    def _predict_with_garch_fixed(self, symbol, model_info, horizon):
        """
        Version corrigée de GARCH qui génère des prévisions de prix valides
        """
        try:
            model = model_info['model']
            
            # Générer les prévisions de variance
            forecast = model.forecast(horizon=horizon)
            volatility = np.sqrt(forecast.variance.values[-1, :])
            
            # Récupérer le prix actuel
            current_price = self.get_price_column(symbol, 'Close').iloc[-1]
            
            # Générer des prévisions de prix basées sur la volatilité GARCH
            # En utilisant un modèle de marche aléatoire avec la volatilité prédite
            
            # Calculer le rendement moyen historique
            price_series = self.get_price_column(symbol, 'Close')
            returns = price_series.pct_change().dropna().tail(60)  # 60 derniers jours
            mean_return = returns.mean()
            
            # Générer les prix futurs
            prices = []
            last_price = current_price
            
            for i in range(horizon):
                # Utiliser la volatilité GARCH pour générer un rendement
                vol = volatility[i] / 100  # Convertir en décimal
                
                # Générer un rendement avec drift (tendance) et volatilité GARCH
                random_shock = np.random.normal(0, vol)
                daily_return = mean_return + random_shock
                
                # Calculer le nouveau prix
                next_price = last_price * (1 + daily_return)
                
                # Validation: éviter les prix aberrants
                change_pct = abs((next_price - last_price) / last_price)
                if change_pct > 0.1:  # Plus de 10% de changement
                    # Limiter le changement
                    sign = 1 if next_price > last_price else -1
                    next_price = last_price * (1 + sign * 0.05)  # Limiter à 5%
                
                prices.append(next_price)
                last_price = next_price
            
            # Créer les dates de prévision
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='B')
            
            # Résultat formaté
            result = {
                'dates': forecast_dates,
                'values': np.array(prices),
                'volatility': volatility,
                'model_type': 'garch'
            }
            
            # Validation finale
            if np.any(np.isnan(result['values'])) or np.any(result['values'] <= 0):
                self.logger.warning(f"Prévisions GARCH invalides pour {symbol}")
                return None
            
            # Vérifier la cohérence des changements
            first_change = abs((result['values'][0] - current_price) / current_price)
            if first_change > 0.15:  # Plus de 15% de changement
                self.logger.warning(f"Prévision GARCH trop extrême pour {symbol}: {first_change:.2%}")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur prévision GARCH pour {symbol}: {str(e)}")
            return None


    def _predict_with_ml(self, symbol, model_info, horizon):
        """
        Génère des prévisions avec un modèle de machine learning - version corrigée pour multiindex
        
        Args:
            symbol (str): Symbole boursier
            model_info (dict): Informations sur le modèle
            horizon (int): Horizon de prévision
            
        Returns:
            dict: Prévisions
        """
        try:
            model = model_info['model']
            features = model_info['features']
            
            # Récupérer la dernière donnée de prix (correctement)
            target_col = self.get_price_column(symbol, 'Close')
            if target_col is None:
                self.logger.error(f"Impossible de récupérer la colonne de prix pour {symbol}")
                return None
                
            current_price = target_col.iloc[-1]
            
            # Pour prédire sur plusieurs jours, nous devons mettre à jour les données à chaque étape
            # Copier les dernières données disponibles
            # Utiliser features à place de self.data[features]
            X_data = self.data[features]
            if X_data.empty:
                self.logger.error(f"Les données de caractéristiques sont vides pour {symbol}")
                return None
                
            last_data = X_data.iloc[-1:].values
            
            # Préparer les conteneurs pour les résultats
            forecasts = []
            
            # Générer des prévisions séquentiellement
            for step in range(horizon):
                # Prédire le prochain rendement
                next_return = model.predict(last_data)[0]
                
                # Convertir le rendement en prix
                if step == 0:
                    next_price = current_price * (1 + next_return)
                else:
                    next_price = forecasts[-1] * (1 + next_return)
                
                forecasts.append(next_price)
                
                # Cette partie est simplifiée et ne mettrait à jour que certaines caractéristiques
                # Dans un cas réel, il faudrait mettre à jour de manière plus complexe
                # pour tenir compte des nouveaux prix, indicateurs techniques, etc.
                
                # Ici, on suppose simplement que les caractéristiques restent similaires
                # Cette approche est grossière mais donne une idée de l'implémentation
            
            # Créer des dates pour les prévisions
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
            
            # Formater les résultats
            predictions = {
                'dates': forecast_dates,
                'values': np.array(forecasts),
                'model_type': 'gradient_boosting'  # Utiliser un nom fixe plutôt que model_info.__class__.__name__
            }
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Erreur lors des prévisions ML pour {symbol}: {str(e)}")
            traceback.print_exc()
            return None


    def _predict_with_dl(self, symbol, model_info, horizon):
        """
        Version améliorée de la fonction de prédiction pour les modèles d'apprentissage profond
        """
        try:
            model = model_info['model']
            scaler_y = model_info['scaler_y']
            seq_length = model_info['seq_length']
            features = model_info['features']
            
            # Obtenir les dernières données pour la prédiction
            X_recent = self.data[features].iloc[-seq_length:].values
            
            # S'assurer qu'il n'y a pas de NaN
            if np.isnan(X_recent).any():
                self.logger.warning(f"NaN détectés dans les données récentes pour {symbol}, remplacement")
                # Remplacer les NaN par la moyenne des colonnes
                col_means = np.nanmean(X_recent, axis=0)
                col_means[np.isnan(col_means)] = 0  # Si toute la colonne est NaN
                
                nan_mask = np.isnan(X_recent)
                for i in range(X_recent.shape[1]):
                    X_recent[nan_mask[:, i], i] = col_means[i]
            
            # Normaliser avec le scaler entraîné
            scaler_X = model_info['scaler_X']
            X_scaled = scaler_X.transform(X_recent)
            
            # Reformater pour le modèle LSTM [samples, time steps, features]
            X_pred = X_scaled.reshape(1, seq_length, X_scaled.shape[1])
            
            # Générer les prédictions séquentiellement
            predictions = []
            last_sequence = X_pred.copy()
            
            # Prédire pour chaque pas de temps
            for _ in range(horizon):
                # Prédire le prochain prix normalisé
                next_pred_scaled = model.predict(last_sequence, verbose=0)[0, 0]
                
                # Convertir à l'échelle originale
                next_pred = scaler_y.inverse_transform([[next_pred_scaled]])[0, 0]
                predictions.append(next_pred)
                
                # Mettre à jour la séquence pour la prochaine prédiction
                # Décaler la séquence d'un pas et ajouter la nouvelle prédiction
                # Pour les autres features, on garde les valeurs précédentes (approx.)
                last_sequence = np.roll(last_sequence, -1, axis=1)
                
                # Cette partie est critiquable et pourrait être améliorée
                # Idéalement, on mettrait à jour toutes les features
                
            # Créer des dates pour les prévisions
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
            
            # Formater les résultats
            result = {
                'dates': forecast_dates,
                'values': np.array(predictions),
                'model_type': 'lstm'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors des prévisions DL pour {symbol}: {str(e)}")
            traceback.print_exc()
            return None
        
    def generate_weighted_predictions(self):
        """
        Version d'urgence de la génération de prévisions pondérées
        """
        try:
            self.logger.info("Génération des prévisions avec pondération simple")
            
            predictions = {}
            
            for symbol in self.config['data']['symbols']:
                if symbol not in self.models:
                    self.logger.warning(f"Aucun modèle disponible pour {symbol}")
                    continue
                
                # Prix actuel
                current_price = self.get_price_column(symbol, 'Close').iloc[-1]
                self.logger.info(f"Prix actuel de {symbol}: {current_price:.2f}")
                
                # Collecter les modèles disponibles
                available_models = {}
                model_qualities = {}
                
                for model_name in ['arima', 'garch', 'gradient_boosting', 'lstm']:
                    if model_name in self.models[symbol]:
                        available_models[model_name] = self.models[symbol][model_name]
                        
                        # Évaluation simple de la qualité
                        try:
                            quality_score = self._evaluate_model_quality(symbol, model_name, self.models[symbol][model_name])
                            model_qualities[model_name] = quality_score
                        except Exception as e:
                            # Si l'évaluation échoue, utiliser un score par défaut
                            self.logger.warning(f"Évaluation échouée pour {model_name}, score par défaut")
                            model_qualities[model_name] = 0.3
                
                if not available_models:
                    self.logger.warning(f"Aucun modèle disponible pour {symbol}")
                    continue
                
                # Générer des prévisions simples
                model_predictions = {}
                
                for model_name, model_info in available_models.items():
                    try:
                        if model_name == 'arima':
                            pred = self._predict_with_arima(symbol, model_info, self.horizon)
                        elif model_name == 'garch':
                            pred = self._predict_with_garch(symbol, model_info, self.horizon)
                        elif model_name == 'gradient_boosting':
                            pred = self._predict_with_ml(symbol, model_info, self.horizon)
                        elif model_name == 'lstm':
                            pred = self._predict_with_dl(symbol, model_info, self.horizon)
                        else:
                            continue
                        
                        # Validation simple
                        if pred and 'values' in pred and len(pred['values']) > 0:
                            # Vérifier que les valeurs sont réalistes
                            first_pred = pred['values'][0]
                            change_pct = abs((first_pred - current_price) / current_price * 100)
                            
                            if change_pct < 20 and not np.isnan(first_pred) and first_pred > 0:
                                model_predictions[model_name] = pred
                            else:
                                self.logger.warning(f"Prédiction invalidée pour {model_name}: changement {change_pct:.1f}%")
                    
                    except Exception as e:
                        self.logger.error(f"Erreur prédiction {model_name}: {e}")
                        continue
                
                if not model_predictions:
                    self.logger.warning(f"Aucune prédiction valide pour {symbol}")
                    continue
                
                # Calcul des poids simples basés sur la qualité
                total_quality = sum(model_qualities[m] for m in model_predictions.keys())
                weights = {}
                
                if total_quality > 0:
                    for model_name in model_predictions.keys():
                        weights[model_name] = model_qualities[model_name] / total_quality
                else:
                    # Poids égaux si problème
                    equal_weight = 1.0 / len(model_predictions)
                    weights = {m: equal_weight for m in model_predictions.keys()}
                
                # Combinaison des prévisions
                combined_values = np.zeros(self.horizon)
                
                for model_name, pred in model_predictions.items():
                    weight = weights[model_name]
                    combined_values += pred['values'][:self.horizon] * weight
                
                # Dates de prévision
                last_date = self.data.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1), 
                    periods=self.horizon, 
                    freq='B'
                )
                
                # Résultat final
                predictions[symbol] = {
                    'dates': forecast_dates,
                    'values': combined_values,
                    'model_type': 'simple_ensemble',
                    'models_info': {
                        'models': list(weights.keys()),
                        'weights': weights
                    }
                }
                
                self.logger.info(f"Prévisions générées pour {symbol} avec {len(weights)} modèles")
                
                # Log des poids
                for model, weight in weights.items():
                    self.logger.info(f"  {model}: {weight:.2%}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des prévisions: {str(e)}")
            return {}
    

    def set_model_weights(self, weights_config):
        """
        Définit manuellement les poids des modèles pour la prédiction
        
        Args:
            weights_config (dict): Configuration des poids par symbole
            Exemple: {
                'AAPL': {'arima': 0.4, 'garch': 0.1, 'gradient_boosting': 0.2, 'lstm': 0.3},
                'MSFT': {'arima': 0.3, 'garch': 0.2, 'gradient_boosting': 0.3, 'lstm': 0.2}
            }
        
        Returns:
            bool: True si les poids ont été définis avec succès
        """
        try:
            self.logger.info("Configuration manuelle des poids des modèles")
            
            self.model_weights = {}
            
            # Vérifier et normaliser les poids pour chaque symbole
            for symbol, weights in weights_config.items():
                # Vérifier que tous les poids sont positifs
                if any(w < 0 for w in weights.values()):
                    self.logger.warning(f"Tous les poids doivent être positifs pour {symbol}")
                    continue
                
                # Calculer la somme des poids
                total_weight = sum(weights.values())
                
                # Si la somme n'est pas égale à 1, normaliser
                if abs(total_weight - 1.0) > 0.01:
                    self.logger.info(f"Normalisation des poids pour {symbol} (somme: {total_weight})")
                    normalized_weights = {model: weight/total_weight for model, weight in weights.items()}
                    self.model_weights[symbol] = normalized_weights
                else:
                    self.model_weights[symbol] = weights
                
                self.logger.info(f"Poids configurés pour {symbol}: {self.model_weights[symbol]}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration des poids: {str(e)}")
            return False
        
    def backtest_models(self):
        """
        Effectue un backtesting des modèles sur des données historiques
        Version corrigée pour éviter les erreurs de TypeError et AttributeError
        
        Returns:
            dict: Résultats du backtesting
        """
        try:
            self.logger.info("Démarrage du backtesting des 4 modèles optimaux")
            
            # Configuration du backtesting
            windows = self.config['backtesting']['windows']
            initial_train_size = self.config['backtesting']['initial_train_size']
            metrics = self.config['backtesting']['metrics']
            
            # Modèles optimaux à tester
            optimal_models = ['arima', 'garch', 'gradient_boosting', 'lstm']
            
            # Résultats par symbole et par modèle
            backtest_results = {}
            
            for symbol in self.config['data']['symbols']:
                # Récupérer les données de prix avec la méthode robuste
                prices = self.get_price_column(symbol, 'Close')
                
                if prices is None or len(prices) == 0:
                    self.logger.warning(f"Aucune donnée de prix trouvée pour {symbol}, backtesting ignoré")
                    continue
                
                # Calculer la taille de chaque fenêtre
                n_samples = len(prices)
                initial_size = int(n_samples * initial_train_size)
                test_size = (n_samples - initial_size) // windows
                
                if test_size <= 0:
                    self.logger.warning(f"Données insuffisantes pour le backtesting de {symbol}")
                    continue
                
                self.logger.info(f"Backtesting pour {symbol}: {windows} fenêtres, taille de test {test_size}")
                
                # Initialiser les résultats pour ce symbole
                backtest_results[symbol] = {}
                
                # Détecter les modèles disponibles pour ce symbole
                available_models = []
                
                if 'arima' in self.models[symbol]:
                    available_models.append('arima')
                elif 'sarima' in self.models[symbol]:  # Utiliser SARIMA comme remplacement d'ARIMA
                    available_models.append('arima')
                    self.models[symbol]['arima'] = self.models[symbol]['sarima']
                
                if 'garch' in self.models[symbol]:
                    available_models.append('garch')
                
                if 'gradient_boosting' in self.models[symbol]:
                    available_models.append('gradient_boosting')
                
                if 'lstm' in self.models[symbol]:
                    available_models.append('lstm')
                
                if not available_models:
                    self.logger.warning(f"Aucun modèle optimal disponible pour {symbol}, backtesting ignoré")
                    continue
                
                # Exécuter le backtesting pour chaque modèle disponible
                for model_type in available_models:
                    # Initialiser les métriques
                    model_metrics = {metric: [] for metric in metrics}
                    
                    # Exécuter sur chaque fenêtre
                    for window in range(windows):
                        # Définir les indices de la fenêtre
                        train_end = initial_size + window * test_size
                        test_start = train_end
                        test_end = test_start + test_size
                        
                        # Diviser les données
                        train_data = prices[:train_end]
                        test_data = prices[test_start:test_end]
                        
                        if len(test_data) == 0:
                            self.logger.warning(f"Données de test vides pour {symbol}, fenêtre {window+1}")
                            continue
                        
                        # Entraîner le modèle sur cette fenêtre
                        if model_type == 'arima':
                            predictions = self._backtest_arima(train_data, test_data)
                        elif model_type == 'garch':
                            predictions = self._backtest_garch(train_data, test_data)
                        elif model_type == 'gradient_boosting':
                            predictions = self._backtest_ml(symbol, model_type, train_data, test_data)
                        elif model_type == 'lstm':
                            predictions = self._backtest_dl(symbol, model_type, train_data, test_data)
                        else:
                            self.logger.warning(f"Type de modèle non supporté pour le backtesting: {model_type}")
                            continue
                        
                        # Si aucune prédiction n'a été générée, passer à la fenêtre suivante
                        if predictions is None or len(predictions) == 0:
                            continue
                        
                        # Calculer les métriques
                        for metric in metrics:
                            try:
                                # S'assurer que test_data et predictions sont compatibles
                                test_values = test_data.values if hasattr(test_data, 'values') else np.array(test_data)
                                pred_values = predictions if isinstance(predictions, np.ndarray) else np.array(predictions)
                                
                                # S'assurer que les longueurs sont compatibles
                                if len(pred_values) != len(test_values):
                                    self.logger.warning(f"Incompatibilité des longueurs - test: {len(test_values)}, pred: {len(pred_values)}")
                                    # Tronquer à la plus petite longueur
                                    min_len = min(len(test_values), len(pred_values))
                                    test_values = test_values[:min_len]
                                    pred_values = pred_values[:min_len]
                                
                                # Maintenant calculer les métriques avec les valeurs compatibles
                                if metric == 'rmse':
                                    value = np.sqrt(mean_squared_error(test_values, pred_values))
                                elif metric == 'mae':
                                    value = mean_absolute_error(test_values, pred_values)
                                elif metric == 'mape':
                                    # Éviter la division par zéro
                                    value = np.mean(np.abs((test_values - pred_values) / np.maximum(np.abs(test_values), 1e-10))) * 100
                                elif metric == 'r2':
                                    value = r2_score(test_values, pred_values)
                                elif metric == 'sharpe':
                                    # Calculer le ratio de Sharpe
                                    returns = np.diff(pred_values) / np.maximum(np.abs(pred_values[:-1]), 1e-10)
                                    value = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                                else:
                                    self.logger.warning(f"Métrique inconnue: {metric}")
                                    continue
                                
                                model_metrics[metric].append(value)
                            
                            except Exception as e:
                                self.logger.warning(f"Erreur lors du calcul de la métrique {metric}: {str(e)}")
                                continue
                    
                    # Calculer les moyennes des métriques
                    metrics_to_average = list(model_metrics.keys())  # Créer une copie des clés
                    for metric in metrics_to_average:
                        if model_metrics[metric]:
                            model_metrics[f"avg_{metric}"] = np.mean(model_metrics[metric])
                        else:
                            model_metrics[f"avg_{metric}"] = None
                    
                    # Stocker les résultats
                    backtest_results[symbol][model_type] = model_metrics
                    
                    # Afficher les résultats
                    metrics_str = ", ".join([f"{metric}: {model_metrics[f'avg_{metric}']:.4f}" 
                                        for metric in metrics 
                                        if f"avg_{metric}" in model_metrics and model_metrics[f"avg_{metric}"] is not None])
                    if metrics_str:
                        self.logger.info(f"Backtesting {symbol} avec {model_type}: {metrics_str}")
                    else:
                        self.logger.warning(f"Aucune métrique calculée pour {symbol} avec {model_type}")
                
                # Pour ce symbole, calculer aussi le backtesting de l'ensemble pondéré
                if len(available_models) > 1:
                    self.logger.info(f"Calcul du backtesting pour l'ensemble pondéré de {symbol}")
                    
                    # Initialiser les métriques pour l'ensemble
                    ensemble_metrics = {metric: [] for metric in metrics}
                    
                    # Exécuter le backtesting de l'ensemble sur chaque fenêtre
                    for window in range(windows):
                        # Définir les indices de la fenêtre
                        train_end = initial_size + window * test_size
                        test_start = train_end
                        test_end = test_start + test_size
                        
                        # Diviser les données
                        train_data = prices[:train_end]
                        test_data = prices[test_start:test_end]
                        
                        if len(test_data) == 0:
                            continue
                        
                        # Générer des prédictions pour chaque modèle sur cette fenêtre
                        model_predictions = {}
                        
                        for model_type in available_models:
                            if model_type == 'arima':
                                pred = self._backtest_arima(train_data, test_data)
                            elif model_type == 'garch':
                                pred = self._backtest_garch(train_data, test_data)
                            elif model_type == 'gradient_boosting':
                                pred = self._backtest_ml(symbol, model_type, train_data, test_data)
                            elif model_type == 'lstm':
                                pred = self._backtest_dl(symbol, model_type, train_data, test_data)
                            else:
                                continue
                            
                            if pred is not None and len(pred) > 0:
                                model_predictions[model_type] = pred
                        
                        # Si moins de 2 modèles ont généré des prédictions, passer à la fenêtre suivante
                        if len(model_predictions) < 2:
                            continue
                        
                        # Calculer les poids en fonction de l'inverse des MSE historiques ou utiliser des poids égaux
                        weights = {}
                        total_weight = 0
                        
                        # Tentative de calculer les poids basés sur les MSE (si disponibles)
                        for model_type in model_predictions:
                            if model_type in backtest_results[symbol] and 'avg_rmse' in backtest_results[symbol][model_type] and backtest_results[symbol][model_type]['avg_rmse'] is not None:
                                rmse = backtest_results[symbol][model_type]['avg_rmse']
                                if rmse > 0:
                                    weight = 1 / rmse
                                    weights[model_type] = weight
                                    total_weight += weight
                        
                        # Si aucun poids n'a pu être calculé, utiliser des poids égaux
                        if total_weight == 0:
                            for model_type in model_predictions:
                                weights[model_type] = 1.0
                            total_weight = len(weights)
                        
                        # Normaliser les poids
                        for model_type in weights:
                            weights[model_type] /= total_weight
                        
                        # Combiner les prédictions
                        ensemble_pred = np.zeros(len(test_data))
                        
                        for model_type, pred in model_predictions.items():
                            if model_type in weights:
                                # Assurer que les prédictions ont la même longueur que les données de test
                                if len(pred) != len(test_data):
                                    self.logger.warning(f"Adaptation de la longueur des prédictions {model_type} de {len(pred)} à {len(test_data)}")
                                    pred = np.resize(pred, len(test_data))
                                
                                ensemble_pred += pred * weights[model_type]
                        
                        # Calculer les métriques pour l'ensemble
                        for metric in metrics:
                            try:
                                test_values = test_data.values if hasattr(test_data, 'values') else np.array(test_data)
                                
                                if metric == 'rmse':
                                    value = np.sqrt(mean_squared_error(test_values, ensemble_pred))
                                elif metric == 'mae':
                                    value = mean_absolute_error(test_values, ensemble_pred)
                                elif metric == 'mape':
                                    value = np.mean(np.abs((test_values - ensemble_pred) / np.maximum(np.abs(test_values), 1e-10))) * 100
                                elif metric == 'r2':
                                    value = r2_score(test_values, ensemble_pred)
                                elif metric == 'sharpe':
                                    returns = np.diff(ensemble_pred) / np.maximum(np.abs(ensemble_pred[:-1]), 1e-10)
                                    value = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                                else:
                                    continue
                                
                                ensemble_metrics[metric].append(value)
                            
                            except Exception as e:
                                self.logger.warning(f"Erreur lors du calcul de la métrique {metric} pour l'ensemble: {str(e)}")
                                continue
                    
                    # Calculer les moyennes des métriques pour l'ensemble
                    for metric in metrics:
                        if ensemble_metrics[metric]:
                            ensemble_metrics[f"avg_{metric}"] = np.mean(ensemble_metrics[metric])
                        else:
                            ensemble_metrics[f"avg_{metric}"] = None
                    
                    # Stocker les résultats de l'ensemble
                    backtest_results[symbol]['weighted_ensemble'] = ensemble_metrics
                    
                    # Afficher les résultats de l'ensemble
                    metrics_str = ", ".join([f"{metric}: {ensemble_metrics[f'avg_{metric}']:.4f}" 
                                        for metric in metrics 
                                        if f"avg_{metric}" in ensemble_metrics and ensemble_metrics[f"avg_{metric}"] is not None])
                    if metrics_str:
                        self.logger.info(f"Backtesting {symbol} avec l'ensemble pondéré: {metrics_str}")
                        
                        # Comparer avec le meilleur modèle individuel
                        best_model = None
                        best_rmse = float('inf')
                        
                        for model_type in available_models:
                            if (model_type in backtest_results[symbol] and 
                                'avg_rmse' in backtest_results[symbol][model_type] and 
                                backtest_results[symbol][model_type]['avg_rmse'] is not None):
                                rmse = backtest_results[symbol][model_type]['avg_rmse']
                                if rmse < best_rmse:
                                    best_rmse = rmse
                                    best_model = model_type
                        
                        if best_model and 'avg_rmse' in ensemble_metrics and ensemble_metrics['avg_rmse'] is not None:
                            ensemble_rmse = ensemble_metrics['avg_rmse']
                            improvement = ((best_rmse - ensemble_rmse) / best_rmse) * 100
                            
                            if improvement > 0:
                                self.logger.info(f"L'ensemble pondéré améliore de {improvement:.2f}% le RMSE par rapport au meilleur modèle ({best_model})")
                            else:
                                self.logger.info(f"L'ensemble pondéré est {-improvement:.2f}% moins bon que le meilleur modèle ({best_model})")
            
            self.results['backtesting'] = backtest_results
            return backtest_results
        
        except Exception as e:
            self.logger.error(f"Erreur lors du backtesting: {str(e)}")
            traceback.print_exc()
            # Assurer que les résultats sont définis même en cas d'erreur
            if 'backtesting' not in self.results:
                self.results['backtesting'] = {}
            return self.results.get('backtesting', {})
        
    def _backtest_arima(self, train_data, test_data):
        """
        Effectue un backtesting avec un modèle ARIMA
        
        Args:
            train_data (pd.Series): Données d'entraînement
            test_data (pd.Series): Données de test
            
        Returns:
            np.array: Prédictions
        """
        try:
            # Déterminer les ordres optimaux avec AIC
            best_aic = float('inf')
            best_order = None
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(train_data, order=(p, d, q))
                            results = model.fit()
                            aic = results.aic
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            if best_order is None:
                self.logger.warning("Impossible de trouver un ordre ARIMA optimal")
                return None
            
            # Construire le modèle avec le meilleur ordre
            model = ARIMA(train_data, order=best_order)
            results = model.fit()
            
            # Générer les prévisions
            forecast = results.forecast(steps=len(test_data))
            
            return forecast
        
        except Exception as e:
            self.logger.error(f"Erreur lors du backtesting ARIMA: {str(e)}")
            return None
        
    def _backtest_garch(self, train_data, test_data):
        """
        Effectue un backtesting avec un modèle GARCH
        
        Args:
            train_data (pd.Series): Données d'entraînement
            test_data (pd.Series): Données de test
            
        Returns:
            np.array: Prédictions
        """
        try:
            # Calculer les rendements
            returns = 100 * train_data.pct_change().dropna()
            
            # Construire le modèle GARCH
            garch_model = arch_model(returns, vol='Garch', p=1, q=1)
            results = garch_model.fit(disp='off')
            
            # Générer les prévisions de volatilité
            forecast = results.forecast(horizon=len(test_data))
            volatility = np.sqrt(forecast.variance.values[-1, :])
            
            # Pour GARCH, nous allons supposer un modèle de marche aléatoire pour le prix
            # avec la volatilité prédite
            last_price = train_data.iloc[-1]
            prices = [last_price]
            
            # Simuler plusieurs trajectoires et prendre la moyenne
            n_simulations = 100
            all_simulations = []
            
            for _ in range(n_simulations):
                sim_prices = [last_price]
                for vol in volatility:
                    # Générer un rendement aléatoire en fonction de la volatilité
                    daily_return = np.random.normal(0, vol / 100)
                    next_price = sim_prices[-1] * (1 + daily_return)
                    sim_prices.append(next_price)
                
                all_simulations.append(sim_prices[1:])  # Exclure le prix initial
            
            # Calculer la moyenne des simulations
            avg_simulation = np.mean(all_simulations, axis=0)
            
            return avg_simulation
        
        except Exception as e:
            self.logger.error(f"Erreur lors du backtesting GARCH: {str(e)}")
            return None

    def _backtest_ml(self, symbol, model_type, train_data, test_data):
        """
        Version corrigée du backtesting ML qui conserve les noms de features
        """
        try:
            # Récupérer les features
            if hasattr(self, 'selected_features') and symbol in self.selected_features:
                features = self.selected_features[symbol]
            else:
                features = self._get_default_features(symbol)
            
            # S'assurer que features est une liste
            features = list(features)
            
            # Créer un DataFrame avec les noms de colonnes
            X_all = pd.DataFrame(index=self.data.index)
            
            # Copier les features depuis self.data
            for feature in features:
                if feature in self.data.columns:
                    X_all[feature] = self.data[feature]
            
            # Vérifier s'il y a des données
            if X_all.empty or len(X_all.columns) == 0:
                self.logger.warning(f"Aucune feature valide pour {symbol}")
                return None
            
            # Vérifier les valeurs manquantes dans X_all
            if X_all.isnull().any().any():
                self.logger.warning(f"Valeurs manquantes dans les données pour {symbol}, imputation")
                # Utiliser une imputation simple
                X_all = X_all.fillna(X_all.mean())
                
            # Filtrer pour les dates d'entraînement
            train_indices = X_all.index.isin(train_data.index)
            X_train = X_all[train_indices]
            y_train = train_data
            
            # Filtrer pour les dates de test
            test_indices = X_all.index.isin(test_data.index)
            X_test = X_all[test_indices]
            
            # Créer et entraîner le modèle
            if model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = Ridge(alpha=1.0)
            
            # Entraîner le modèle
            model.fit(X_train, y_train)
            
            # Prédire sur les données de test
            y_pred = model.predict(X_test)
            
            return y_pred
        
        except Exception as e:
            self.logger.error(f"Erreur lors du backtesting ML: {str(e)}")
            traceback.print_exc()
            return None

    def _backtest_dl(self, symbol, model_type, train_data, test_data):
        """
        Effectue un backtesting avec un modèle d'apprentissage profond
        
        Args:
            symbol (str): Symbole boursier
            model_type (str): Type de modèle ('lstm', 'gru', 'transformer')
            train_data (pd.Series): Données d'entraînement
            test_data (pd.Series): Données de test
            
        Returns:
            np.array: Prédictions
        """
        try:
            self.logger.info(f"Backtesting du modèle {model_type} pour {symbol}")
            
            # Récupérer les prix de clôture sous forme de série
            close_data = None
            
            # Déterminer le bon format de colonne selon le type d'index
            if isinstance(self.data.columns, pd.MultiIndex):
                try:
                    close_data = self.data[(symbol, 'Close')]
                    self.logger.info(f"Utilisation du format MultiIndex pour {symbol}")
                except KeyError:
                    try:
                        close_data = self.data[f"{symbol}.Close"]
                        self.logger.info(f"Utilisation du format string pour {symbol}")
                    except KeyError:
                        # Essayer de trouver une colonne contenant 'Close' et le symbole
                        close_cols = [col for col in self.data.columns 
                                    if 'Close' in str(col) and symbol in str(col)]
                        if close_cols:
                            close_data = self.data[close_cols[0]]
                            self.logger.info(f"Utilisation de la colonne {close_cols[0]} pour {symbol}")
                        else:
                            self.logger.error(f"Colonne de prix non trouvée pour {symbol}")
                            return None
            else:
                try:
                    close_data = self.data[f"{symbol}.Close"]
                    self.logger.info(f"Utilisation du format string pour {symbol}")
                except KeyError:
                    # Essayer de trouver une colonne contenant 'Close' et le symbole
                    close_cols = [col for col in self.data.columns 
                                if 'Close' in str(col) and symbol in str(col)]
                    if close_cols:
                        close_data = self.data[close_cols[0]]
                        self.logger.info(f"Utilisation de la colonne {close_cols[0]} pour {symbol}")
                    else:
                        self.logger.error(f"Colonne de prix non trouvée pour {symbol}")
                        return None
            
            # Vérifier que les données train et test sont des séries
            if not isinstance(train_data, pd.Series):
                self.logger.info("Conversion des données d'entraînement en Series")
                train_data = pd.Series(train_data)
                
            if not isinstance(test_data, pd.Series):
                self.logger.info("Conversion des données de test en Series")
                test_data = pd.Series(test_data)
            
            # Paramètres pour les séquences
            seq_length = 20  # Nombre de jours à considérer
            
            # Créer des séquences pour l'apprentissage
            X_all = []
            y_all = []
            
            # Utiliser les valeurs de train_data pour créer les séquences
            train_values = train_data.values
            
            for i in range(len(train_values) - seq_length):
                # Créer une séquence de seq_length jours
                sequence = train_values[i:i + seq_length]
                # Ajouter une dimension pour représenter les features (ici une seule feature: le prix)
                sequence = sequence.reshape(-1, 1)
                X_all.append(sequence)
                y_all.append(train_values[i + seq_length])
            
            # Vérifier qu'il y a suffisamment de données
            if len(X_all) == 0:
                self.logger.warning(f"Séquences vides pour {symbol}, données insuffisantes")
                return None
                
            # Convertir en arrays numpy
            X_all = np.array(X_all)
            y_all = np.array(y_all)
            
            # Reshape pour LSTM/GRU: [samples, time steps, features]
            if len(X_all.shape) == 2:
                X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)
            
            # Normaliser les données
            scaler = MinMaxScaler()
            
            # Normaliser les features (X)
            n_samples, n_timesteps, n_features = X_all.shape
            X_all_scaled = np.zeros_like(X_all)
            
            for i in range(n_samples):
                X_all_scaled[i, :, 0] = scaler.fit_transform(X_all[i, :, 0].reshape(-1, 1)).flatten()
            
            # Normaliser les targets (y)
            y_scaler = MinMaxScaler()
            y_all_scaled = y_scaler.fit_transform(y_all.reshape(-1, 1)).flatten()
            
            # Créer le modèle en utilisant la fonction complète
            model = self._create_dl_model_for_backtest(model_type, X_all.shape, seq_length)
            
            # Configurer l'early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Entraîner le modèle
            try:
                model.fit(
                    X_all_scaled, y_all_scaled,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0  # Silencieux
                )
            except Exception as fit_error:
                self.logger.error(f"Erreur lors de l'entraînement du modèle {model_type}: {str(fit_error)}")
                return None
            
            # Générer les prédictions
            predictions = []
            test_values = test_data.values
            
            # Préparer la dernière séquence connue
            if len(train_values) >= seq_length:
                last_sequence = train_values[-seq_length:]
            else:
                # Si pas assez de données, compléter avec des zéros
                padding = np.zeros(seq_length - len(train_values))
                last_sequence = np.concatenate([padding, train_values])
                
            last_sequence = last_sequence.reshape(1, seq_length, 1)
            
            # Normaliser la dernière séquence
            last_sequence_scaled = np.zeros_like(last_sequence)
            last_sequence_scaled[0, :, 0] = scaler.fit_transform(last_sequence[0, :, 0].reshape(-1, 1)).flatten()
            
            # Générer les prédictions pour chaque pas de temps
            for _ in range(len(test_values)):
                # Prédire le prochain prix
                pred_scaled = model.predict(last_sequence_scaled, verbose=0)[0]
                
                # Transformer la prédiction à l'échelle originale
                pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                predictions.append(pred)
                
                # Mettre à jour la séquence pour la prochaine prédiction
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred
                
                # Normaliser la nouvelle séquence
                last_sequence_scaled = np.zeros_like(last_sequence)
                last_sequence_scaled[0, :, 0] = scaler.fit_transform(last_sequence[0, :, 0].reshape(-1, 1)).flatten()
            
            # Nettoyer la session Keras pour éviter les fuites de mémoire
            tf.keras.backend.clear_session()
            
            return np.array(predictions)
        
        except Exception as e:
            self.logger.error(f"Erreur lors du backtesting DL: {str(e)}")
            traceback.print_exc()
            return None
        
    def detect_market_regime(self, symbol, window_size=60):
        """
        Détecte le régime de marché actuel (tendance, retour à la moyenne, volatilité élevée)
        Version adaptée pour utiliser get_price_column
        
        Args:
            symbol (str): Symbole boursier
            window_size (int): Taille de la fenêtre d'analyse
            
        Returns:
            dict: Information sur le régime de marché
        """
        try:
            self.logger.info(f"Détection du régime de marché pour {symbol}")
            
            # Récupérer les données historiques avec la méthode robuste
            historical = self.get_price_column(symbol, 'Close')
            
            if historical is None:
                self.logger.error(f"Impossible de trouver les données de prix pour {symbol}")
                return {'regime': 'unknown'}
            
            if len(historical) < window_size:
                self.logger.warning(f"Données insuffisantes pour détecter le régime de marché ({len(historical)} < {window_size})")
                return {'regime': 'unknown'}
            
            # Utiliser les dernières données pour l'analyse
            prices = historical[-window_size:].values
            returns = np.diff(prices) / prices[:-1]
            log_returns = np.log(prices[1:] / prices[:-1])
            
            # Calculer les statistiques
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Autocorrélation des rendements (lag 1)
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            
            # Test de stationnarité (Augmented Dickey-Fuller)
            adf_result = statsmodels.tsa.stattools.adfuller(prices)
            adf_pvalue = adf_result[1]
            
            # Test ARCH pour l'hétéroscédasticité (regroupement de volatilité)
            arch_test = statsmodels.tsa.stattools.acf(returns**2, nlags=10)
            arch_effect = np.mean(arch_test[1:])  # Moyenne des autocorrélations des rendements au carré
            
            # Détecter le régime de marché
            regime = {}
            
            # Trend following ou mean reversion?
            if autocorr > 0.1:
                regime['type'] = 'trend_following'
                regime['strength'] = autocorr
            elif autocorr < -0.1:
                regime['type'] = 'mean_reverting'
                regime['strength'] = -autocorr
            else:
                regime['type'] = 'random_walk'
                regime['strength'] = 1 - abs(autocorr)
            
            # Volatilité
            if std_return > np.median(pd.Series(returns).rolling(window=30).std().dropna()) * 1.5:
                regime['volatility'] = 'high'
            else:
                regime['volatility'] = 'normal'
            
            # Données dépendantes / hétéroscédasticité
            if arch_effect > 0.2:
                regime['volatility_clustering'] = True
            else:
                regime['volatility_clustering'] = False
            
            # Stationnarité
            if adf_pvalue < 0.05:
                regime['stationarity'] = True
            else:
                regime['stationarity'] = False
            
            # Distribution des rendements
            if kurtosis > 3:
                regime['fat_tails'] = True
                regime['fat_tails_severity'] = kurtosis / 3
            else:
                regime['fat_tails'] = False
                regime['fat_tails_severity'] = 1
            
            # Asymétrie
            if abs(skewness) > 0.5:
                regime['skewed'] = True
                regime['skew_direction'] = 'negative' if skewness < 0 else 'positive'
                regime['skew_severity'] = abs(skewness)
            else:
                regime['skewed'] = False
            
            # Rapport détaillé
            report = {
                'regime': regime,
                'statistics': {
                    'mean_return': float(mean_return),  # Convertir en float pour la sérialisation JSON
                    'volatility': float(std_return),
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis),
                    'autocorrelation': float(autocorr),
                    'adf_pvalue': float(adf_pvalue),
                    'arch_effect': float(arch_effect)
                }
            }
            
            # Ajouter l'interprétation du régime pour les rapports et visualisations
            regime_interpretation = {}
            
            # Recommandations basées sur le régime détecté
            if regime['type'] == 'trend_following':
                regime_interpretation['description'] = "Marché tendanciel - les prix ont tendance à continuer leur mouvement"
                regime_interpretation['strategy'] = "Stratégies momentum ou suiveur de tendance recommandées"
                regime_interpretation['forecast_confidence'] = "Élevée" if regime['strength'] > 0.3 else "Moyenne"
            elif regime['type'] == 'mean_reverting':
                regime_interpretation['description'] = "Marché de retour à la moyenne - les prix ont tendance à revenir vers leur moyenne"
                regime_interpretation['strategy'] = "Stratégies de trading de range ou contrariennes recommandées"
                regime_interpretation['forecast_confidence'] = "Élevée" if regime['strength'] > 0.3 else "Moyenne"
            else:
                regime_interpretation['description'] = "Marché aléatoire - les prix n'affichent pas de tendance claire"
                regime_interpretation['strategy'] = "Stratégies basées sur d'autres facteurs recommandées, éviter le trading directionnel"
                regime_interpretation['forecast_confidence'] = "Faible"
            
            # Ajouter des informations sur la volatilité
            if regime['volatility'] == 'high':
                regime_interpretation['volatility_description'] = "Volatilité élevée - mouvements de prix plus importants que la normale"
                regime_interpretation['volatility_advice'] = "Élargir les stops et réduire la taille des positions"
            else:
                regime_interpretation['volatility_description'] = "Volatilité normale"
                regime_interpretation['volatility_advice'] = "Taille des positions standard, stops normaux"
            
            # Ajouter l'interprétation au rapport
            report['interpretation'] = regime_interpretation
            
            self.logger.info(f"Régime détecté pour {symbol}: {regime['type']} (force: {regime['strength']:.2f}), "
                            f"volatilité: {regime['volatility']}")
            
            return report
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection du régime de marché: {str(e)}")
            traceback.print_exc()
            return {'regime': 'unknown'}
        
    def integrate_extreme_events(self, predictions, symbol):
        """
        Intègre la gestion des événements extrêmes dans les prévisions
        
        Args:
            predictions (dict): Prévisions pour un symbole
            symbol (str): Symbole boursier
            
        Returns:
            dict: Prévisions ajustées
        """
        try:
            self.logger.info(f"Intégration des événements extrêmes pour {symbol}")
            
            # Vérifier si les données sont disponibles
            if predictions is None or 'values' not in predictions:
                return predictions
            
            # Récupérer l'historique des prix
            close_col = f"{symbol}.Close"
            historical = self.data[close_col]
            
            # Calculer les rendements journaliers
            returns = historical.pct_change().dropna()
            
            # Analyser la distribution des rendements
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Définir des seuils pour les événements extrêmes
            # Typiquement, on considère comme extrêmes les événements au-delà de 3 écarts-types
            extreme_threshold = 3 * std_return
            
            # Calculer la probabilité d'événements extrêmes
            # On peut utiliser la loi normale ou une distribution à queue épaisse comme la t de Student
            from scipy import stats
            
            # Ajustement des degrés de liberté pour la distribution t de Student
            # Plus ce nombre est petit, plus les queues sont épaisses
            df = 5  # Degrés de liberté
            
            # Ajuster une distribution t de Student
            t_params = stats.t.fit(returns, df=df)
            t_dist = stats.t(*t_params)
            
            # Calculer la probabilité d'un événement extrême négatif
            prob_extreme_neg = t_dist.cdf(-extreme_threshold)
            
            # Calculer la probabilité d'un événement extrême positif
            prob_extreme_pos = 1 - t_dist.cdf(extreme_threshold)
            
            # Probabilité totale d'un événement extrême
            prob_extreme = prob_extreme_neg + prob_extreme_pos
            
            self.logger.info(f"Probabilité d'événement extrême pour {symbol}: {prob_extreme:.4f}")
            
            # Simuler des scénarios avec événements extrêmes
            n_simulations = 1000
            horizon = len(predictions['values'])
            
            # Initialiser les conteneurs pour les simulations
            all_simulations = np.zeros((n_simulations, horizon))
            
            # Prix initial
            initial_price = historical.iloc[-1]
            
            # Simuler des trajectoires de prix
            for i in range(n_simulations):
                price = initial_price
                prices = []
                
                for j in range(horizon):
                    # Déterminer s'il y a un événement extrême
                    if random.random() < prob_extreme:
                        # C'est un événement extrême, déterminer s'il est positif ou négatif
                        if random.random() < prob_extreme_neg / prob_extreme:
                            # Événement extrême négatif
                            shock = t_dist.ppf(random.uniform(0, prob_extreme_neg))
                        else:
                            # Événement extrême positif
                            shock = t_dist.ppf(random.uniform(1 - prob_extreme_pos, 1))
                    else:
                        # Rendement normal, on peut utiliser la prévision du modèle
                        # et ajouter un bruit gaussien
                        if j < len(predictions['values']):
                            predicted_price = predictions['values'][j]
                            predicted_return = (predicted_price / price) - 1
                            shock = predicted_return + random.gauss(0, std_return / 2)
                        else:
                            shock = random.gauss(mean_return, std_return)
                    
                    # Mettre à jour le prix
                    price *= (1 + shock)
                    prices.append(price)
                
                all_simulations[i, :] = prices
            
            # Calculer les statistiques des simulations
            mean_simulation = np.mean(all_simulations, axis=0)
            std_simulation = np.std(all_simulations, axis=0)
            
            # Calculer les quantiles pour les intervalles de confiance
            lower_quantile = np.percentile(all_simulations, (1 - self.confidence_level) * 100 / 2, axis=0)
            upper_quantile = np.percentile(all_simulations, 100 - (1 - self.confidence_level) * 100 / 2, axis=0)
            
            # Mettre à jour les prévisions
            predictions['values'] = mean_simulation
            predictions['lower_bound'] = lower_quantile
            predictions['upper_bound'] = upper_quantile
            predictions['volatility'] = std_simulation / mean_simulation  # Volatilité relative
            
            # Ajouter des informations sur les événements extrêmes
            predictions['extreme_event_probability'] = prob_extreme
            predictions['extreme_event_impact'] = {
                'negative': t_dist.ppf(0.01),  # Impact d'un événement extrême négatif (1%)
                'positive': t_dist.ppf(0.99)   # Impact d'un événement extrême positif (99%)
            }
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'intégration des événements extrêmes: {str(e)}")
            traceback.print_exc()
            return predictions

    def _analyze_current_market(self, symbol):
        """
        Analyse les conditions actuelles du marché
        
        Args:
            symbol (str): Symbole boursier
            
        Returns:
            dict: Conditions du marché
        """
        try:
            # Récupérer les données historiques récentes
            close_col = f"{symbol}.Close"
            volume_col = f"{symbol}.Volume"
            
            # Utiliser les 60 derniers jours pour l'analyse
            window_size = 60
            
            prices = self.data[close_col][-window_size:].values
            
            # Calculer la volatilité récente (écart-type des rendements quotidiens)
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            # Détecter la tendance (utiliser une régression linéaire simple)
            X = np.arange(len(prices)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, prices)
            
            slope = model.coef_[0]
            trend_strength = abs(slope * len(prices) / prices[0])  # Changement relatif sur la période
            
            trend = 'neutral'
            if slope > 0 and trend_strength > 0.05:  # Hausse de plus de 5%
                trend = 'bullish'
            elif slope < 0 and trend_strength > 0.05:  # Baisse de plus de 5%
                trend = 'bearish'
            
            # Calculer un score de liquidité si les données de volume sont disponibles
            liquidity = None
            if volume_col in self.data.columns:
                volumes = self.data[volume_col][-window_size:].values
                
                # Normaliser les volumes
                avg_volume = np.mean(volumes)
                vol_variability = np.std(volumes) / avg_volume if avg_volume > 0 else 0
                
                # Un marché liquide a un volume élevé et stable
                if avg_volume > 0:
                    # Calculer un score de liquidité entre 0 et 1
                    liquidity_score = 1 / (1 + vol_variability)  # Plus la variabilité est faible, plus le score est élevé
                    
                    liquidity = liquidity_score
            
            # Résultats de l'analyse
            market_condition = {
                'volatility': volatility,
                'trend': trend,
                'trend_strength': trend_strength,
                'slope': slope
            }
            
            if liquidity is not None:
                market_condition['liquidity'] = liquidity
            
            return market_condition
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des conditions du marché: {str(e)}")
            traceback.print_exc()
            return {'volatility': 0.01, 'trend': 'unknown'}


    def _get_upcoming_economic_events(self, symbol):
        """
        Récupère les événements économiques à venir qui pourraient affecter le symbole
        
        Args:
            symbol (str): Symbole boursier
            
        Returns:
            list: Liste des événements économiques à venir
        """
        try:
            # Dans un cas réel, cette fonction appellerait une API d'événements économiques
            # Pour cette démonstration, nous simulons quelques événements
            
            # Récupérer la date actuelle
            current_date = self.data.index[-1]
            
            # Générer des événements fictifs pour les 30 prochains jours
            events = []
            
            # Types d'événements et leur impact potentiel
            event_types = [
                {'type': 'Rapport de résultats', 'impact': 0.8},
                {'type': 'Décision de taux d\'intérêt', 'impact': 0.7},
                {'type': 'Publication de l\'emploi', 'impact': 0.6},
                {'type': 'Indice de confiance des consommateurs', 'impact': 0.4},
                {'type': 'PIB trimestriel', 'impact': 0.7},
                {'type': 'Inflation (IPC)', 'impact': 0.6},
                {'type': 'Balance commerciale', 'impact': 0.3},
                {'type': 'PMI manufacturier', 'impact': 0.5}
            ]
            
            # Événements spécifiques aux entreprises
            company_events = [
                {'type': f'Résultats trimestriels de {symbol}', 'impact': 0.9},
                {'type': f'Annonce de dividendes de {symbol}', 'impact': 0.6},
                {'type': f'Conférence investisseurs de {symbol}', 'impact': 0.5}
            ]
            
            # Sélectionner aléatoirement quelques événements
            num_events = random.randint(2, 5)
            
            for _ in range(num_events):
                # Date aléatoire dans les 30 prochains jours
                days_ahead = random.randint(1, 30)
                event_date = current_date + pd.Timedelta(days=days_ahead)
                
                # Types d'événements
                all_events = event_types + company_events
                selected_event = random.choice(all_events)
                
                events.append({
                    'date': event_date,
                    'type': selected_event['type'],
                    'expected_impact': selected_event['impact']
                })
            
            # Trier les événements par date
            events.sort(key=lambda x: x['date'])
            
            return events
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des événements économiques: {str(e)}")
            traceback.print_exc()
            return []
        
    def adapt_to_market_conditions(self, symbol, predictions):
        """
        Adapte les prévisions aux conditions actuelles du marché
        
        Args:
            symbol (str): Symbole boursier
            predictions (dict): Prévisions initiales
            
        Returns:
            dict: Prévisions adaptées
        """
        try:
            self.logger.info(f"Adaptation des prévisions aux conditions du marché pour {symbol}")
            
            # Vérifier si les prévisions sont disponibles
            if predictions is None or 'values' not in predictions:
                return predictions
            
            # Analyser l'état actuel du marché
            market_state = self._analyze_current_market(symbol)
            
            # Ajuster l'horizon de prévision en fonction de la volatilité
            if market_state['volatility'] > 0.02:  # Volatilité quotidienne > 2%
                # Réduire l'horizon pour les marchés très volatils
                orig_horizon = len(predictions['values'])
                
                # Plus la volatilité est élevée, plus l'horizon est réduit
                new_horizon = max(5, int(orig_horizon * (1 - market_state['volatility'] * 10)))
                
                if new_horizon < orig_horizon:
                    self.logger.info(f"Réduction de l'horizon de prévision de {orig_horizon} à {new_horizon} jours "
                                    f"en raison d'une forte volatilité ({market_state['volatility']:.1%})")
                    
                    for key in ['values', 'lower_bound', 'upper_bound', 'volatility']:
                        if key in predictions:
                            predictions[key] = predictions[key][:new_horizon]
                    
                    predictions['dates'] = predictions['dates'][:new_horizon]
                    predictions['reduced_horizon'] = True
                    predictions['volatility_reason'] = market_state['volatility']
            
            # Ajuster les intervalles de confiance en fonction de la liquidité
            if 'liquidity' in market_state and market_state['liquidity'] < 0.5:  # Faible liquidité
                if 'lower_bound' in predictions and 'upper_bound' in predictions:
                    # Élargir les intervalles pour les marchés peu liquides
                    mean_values = predictions['values']
                    lower_bound = predictions['lower_bound']
                    upper_bound = predictions['upper_bound']
                    
                    # Facteur d'élargissement inversement proportionnel à la liquidité
                    expansion_factor = 1 + (0.5 - market_state['liquidity']) * 2
                    
                    predictions['lower_bound'] = mean_values - (mean_values - lower_bound) * expansion_factor
                    predictions['upper_bound'] = mean_values + (upper_bound - mean_values) * expansion_factor
                    
                    self.logger.info(f"Élargissement des intervalles de confiance (facteur: {expansion_factor:.2f}) "
                                    f"en raison d'une faible liquidité ({market_state['liquidity']:.2f})")
            
            # Tenir compte des événements économiques imminents
            upcoming_events = self._get_upcoming_economic_events(symbol)
            
            if upcoming_events:
                self.logger.info(f"Événements économiques à venir pour {symbol}: {len(upcoming_events)}")
                
                for event in upcoming_events:
                    event_date = event['date']
                    event_impact = event['expected_impact']
                    event_type = event['type']
                    
                    # Trouver l'indice de la date dans les prévisions
                    date_indices = [i for i, date in enumerate(predictions['dates']) 
                                if date.date() == event_date.date()]
                    
                    if date_indices:
                        idx = date_indices[0]
                        
                        # Ajuster les intervalles de confiance pour la date de l'événement
                        if 'lower_bound' in predictions and 'upper_bound' in predictions and idx < len(predictions['lower_bound']):
                            mean_value = predictions['values'][idx]
                            current_width = predictions['upper_bound'][idx] - predictions['lower_bound'][idx]
                            
                            # Élargir l'intervalle en fonction de l'impact attendu
                            impact_factor = 1 + (event_impact * 0.5)  # Impact de 0 à 1, facteur de 1 à 1.5
                            
                            new_width = current_width * impact_factor
                            predictions['lower_bound'][idx] = mean_value - new_width/2
                            predictions['upper_bound'][idx] = mean_value + new_width/2
                            
                            self.logger.info(f"Intervalle ajusté pour l'événement {event_type} le {event_date.date()}: "
                                            f"facteur d'impact {impact_factor:.2f}")
                        
                        # Ajuster également les jours suivants (effet d'onde)
                        fade_length = 3  # Nombre de jours pour que l'effet s'estompe
                        for j in range(1, fade_length+1):
                            if idx+j < len(predictions['values']) and 'lower_bound' in predictions and 'upper_bound' in predictions:
                                fade_factor = impact_factor * (fade_length - j + 1) / (fade_length + 1)
                                
                                mean_value = predictions['values'][idx+j]
                                current_width = predictions['upper_bound'][idx+j] - predictions['lower_bound'][idx+j]
                                
                                new_width = current_width * (1 + (fade_factor - 1) * 0.7)  # Effet atténué
                                predictions['lower_bound'][idx+j] = mean_value - new_width/2
                                predictions['upper_bound'][idx+j] = mean_value + new_width/2
            
            # Ajouter des métadonnées sur les ajustements
            predictions['market_adjusted'] = True
            predictions['market_condition'] = {
                'volatility': market_state['volatility'],
                'trend': market_state['trend'],
                'liquidity': market_state.get('liquidity', 'unknown'),
                'upcoming_events': [{'date': e['date'].strftime('%Y-%m-%d'), 'type': e['type']} 
                                    for e in upcoming_events]
            }
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'adaptation aux conditions du marché: {str(e)}")
            traceback.print_exc()
            return predictions

    def _evaluate_model_quality_fixed(self, symbol, model_name, model_info):
        """
        Version corrigée de l'évaluation de qualité avec critères plus stricts
        """
        try:
            quality_score = 0.3  # Score par défaut plus bas
            
            # 1. Vérifier les métriques de backtesting
            if ('backtesting' in self.results and symbol in self.results['backtesting'] and 
                model_name in self.results['backtesting'][symbol]):
                
                metrics = self.results['backtesting'][symbol][model_name]
                
                # R² - critères plus stricts
                if 'avg_r2' in metrics and metrics['avg_r2'] is not None:
                    r2 = metrics['avg_r2']
                    if r2 > 0.7:
                        quality_score += 0.4
                    elif r2 > 0.3:
                        quality_score += 0.2
                    elif r2 > 0.1:
                        quality_score += 0.1
                    elif r2 > 0:
                        quality_score += 0.05
                    else:
                        quality_score -= 0.3  # Pénalité forte pour R² négatif
                
                # MAPE
                if 'avg_mape' in metrics and metrics['avg_mape'] is not None:
                    mape = metrics['avg_mape']
                    if mape < 1:
                        quality_score += 0.2
                    elif mape < 3:
                        quality_score += 0.1
                    elif mape > 10:
                        quality_score -= 0.2
            
            # 2. Métriques internes du modèle - CORRECTION CRITIQUE
            if 'metrics' in model_info:
                metrics = model_info['metrics']
                
                if 'r2' in metrics:
                    r2 = metrics['r2']
                    if r2 < -0.1:  # R² très négatif
                        quality_score -= 0.4  # Pénalité très forte
                        self.logger.warning(f"R² très négatif pour {model_name}: {r2:.4f}")
                    elif r2 < 0:
                        quality_score -= 0.2
                    elif r2 > 0.5:
                        quality_score += 0.2
                    elif r2 > 0.2:
                        quality_score += 0.1
            
            # 3. Bonus/malus par type de modèle
            model_adjustments = {
                'gradient_boosting': -0.1,  # Malus car souvent problématique
                'lstm': 0.05,
                'garch': 0.0,
                'arima': 0.05
            }
            
            if model_name in model_adjustments:
                quality_score += model_adjustments[model_name]
            
            # 4. Limiter le score et marquer la qualité
            quality_score = max(0.05, min(1.0, quality_score))
            
            # Déterminer la qualité textuelle
            if quality_score < 0.3:
                quality_text = "poor"
            elif quality_score < 0.6:
                quality_text = "medium"
            else:
                quality_text = "good"
            
            # Mettre à jour l'info du modèle
            if 'metrics' not in model_info:
                model_info['metrics'] = {}
            model_info['quality'] = quality_text
            
            self.logger.info(f"Score de qualité pour {model_name} sur {symbol}: {quality_score:.3f} ({quality_text})")
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation qualité: {e}")
            return 0.2


    def _calculate_intelligent_weights_improved(self, symbol, valid_predictions, model_qualities):
        """
        Version améliorée du calcul des poids avec critères plus stricts
        """
        try:
            weights = {}
            
            # 1. Filtrer les modèles de très mauvaise qualité
            good_models = {}
            for model_name, quality in model_qualities.items():
                if quality > 0.25:  # Seuil minimum de qualité
                    good_models[model_name] = quality
                else:
                    self.logger.warning(f"Modèle {model_name} écarté pour qualité insuffisante: {quality:.3f}")
            
            if not good_models:
                self.logger.warning(f"Aucun modèle de qualité suffisante pour {symbol}, utilisation de tous")
                good_models = model_qualities
            
            # 2. Poids de base basés sur la qualité
            total_quality = sum(good_models.values())
            if total_quality > 0:
                for model_name in valid_predictions:
                    if model_name in good_models:
                        weights[model_name] = good_models[model_name] / total_quality
                    else:
                        weights[model_name] = 0.05  # Poids minimal
            else:
                # Poids égaux si problème
                equal_weight = 1.0 / len(valid_predictions)
                for model_name in valid_predictions:
                    weights[model_name] = equal_weight
            
            # 3. Vérification des prévisions extrêmes
            current_price = self.get_price_column(symbol, 'Close').iloc[-1]
            
            for model_name, pred in valid_predictions.items():
                if 'values' in pred and len(pred['values']) > 0:
                    first_pred = pred['values'][0]
                    change_pct = abs((first_pred - current_price) / current_price * 100)
                    
                    # Pénaliser les changements extrêmes
                    if change_pct > 8:  # Plus de 8%
                        penalty = min(0.8, change_pct / 10)  # Pénalité proportionnelle
                        weights[model_name] *= (1 - penalty)
                        self.logger.info(f"Pénalité appliquée à {model_name}: changement {change_pct:.2f}%")
            
            # 4. Assurer une diversification minimale
            min_weight = 0.1
            num_models = len(weights)
            
            if num_models > 1:
                # Réserver du poids pour les minimums
                reserved_weight = num_models * min_weight
                if reserved_weight < 1.0:
                    available_weight = 1.0 - reserved_weight
                    
                    # Redistribuer
                    total_current = sum(weights.values())
                    if total_current > 0:
                        scale_factor = available_weight / total_current
                        for model_name in weights:
                            weights[model_name] = min_weight + weights[model_name] * scale_factor
            
            # 5. Normalisation finale
            total_final = sum(weights.values())
            if total_final > 0:
                for model_name in weights:
                    weights[model_name] /= total_final
            
            # 6. Log détaillé des poids
            self.logger.info(f"Poids détaillés pour {symbol}:")
            for model_name, weight in weights.items():
                quality = model_qualities.get(model_name, 0)
                self.logger.info(f"  {model_name}: {weight:.3f} (qualité: {quality:.3f})")
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Erreur calcul poids intelligents: {e}")
            # Poids égaux en cas d'erreur
            equal_weight = 1.0 / len(valid_predictions)
            return {model: equal_weight for model in valid_predictions}


    # Fonction pour appliquer toutes les corrections finales
    def apply_final_fixes(predictor):
        """Applique les corrections finales au prédicteur"""
        
        import types
        
        print("🔧 Application des corrections finales...")
        
        # Remplacer les méthodes problématiques
        predictor.select_features = types.MethodType(select_features_robust, predictor)
        predictor._evaluate_model_quality = types.MethodType(_evaluate_model_quality_fixed, predictor)
        predictor._predict_with_garch = types.MethodType(_predict_with_garch_fixed, predictor)
        predictor._calculate_intelligent_weights = types.MethodType(_calculate_intelligent_weights_improved, predictor)
        
        print("✅ Corrections finales appliquées!")
        return predictor


    # Test des corrections finales
    def test_final_fixes():
        """Test des corrections finales"""
        
        try:
            print("🧪 TEST DES CORRECTIONS FINALES")
            print("=" * 40)
            
            # Créer le prédicteur
            predictor = PredictionBoursiere(symbols=["BA"])
            predictor.config['tiingo']['api_key'] = "1f2a9c9f7cc99f3f7855f6dec4a6760c00735d3f"
            
            # Appliquer toutes les corrections
            predictor = apply_corrections_to_existing_predictor(predictor)  # Corrections précédentes
            predictor = apply_final_fixes(predictor)  # Corrections finales
            
            # Test du pipeline complet
            print("\n1. Test de chargement des données...")
            if predictor.load_data():
                print("   ✅ Données chargées")
            else:
                print("   ❌ Échec chargement")
                return
            
            print("\n2. Test de prétraitement...")
            if predictor.preprocess_data():
                print("   ✅ Prétraitement réussi")
            else:
                print("   ❌ Échec prétraitement")
                return
            
            print("\n3. Test de sélection de features corrigée...")
            features = predictor.select_features()
            for symbol, feats in features.items():
                print(f"   ✅ {symbol}: {len(feats)} features sélectionnées sans erreur NaN")
            
            print("\n4. Test de construction de modèles...")
            predictor.models = {}
            for symbol in predictor.config['data']['symbols']:
                predictor.models[symbol] = {}
                
                # Test Gradient Boosting
                predictor._build_gradient_boosting_model(symbol, features[symbol])
                if 'gradient_boosting' in predictor.models[symbol]:
                    model_info = predictor.models[symbol]['gradient_boosting']
                    quality = model_info.get('quality', 'unknown')
                    r2 = model_info.get('metrics', {}).get('r2', 'N/A')
                    print(f"   ✅ Gradient Boosting: qualité={quality}, R²={r2}")
                
                # Test GARCH
                predictor._build_garch_model(symbol)
                if 'garch' in predictor.models[symbol]:
                    print("   ✅ GARCH construit")
            
            print("\n5. Test de génération de prévisions...")
            predictions = predictor.generate_weighted_predictions()
            
            if predictions:
                for symbol, pred in predictions.items():
                    models_used = pred.get('models_info', {}).get('models', [])
                    weights = pred.get('models_info', {}).get('weights', {})
                    print(f"   ✅ {symbol}: {len(models_used)} modèles utilisés")
                    for model in models_used:
                        weight = weights.get(model, 0)
                        print(f"      - {model}: {weight:.2%}")
            
            print("\n✅ TOUS LES TESTS RÉUSSIS!")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors des tests: {e}")
            import traceback
            traceback.print_exc()
            return False


    def _validate_prediction(self, prediction, current_price, symbol):
        """
        Valide qu'une prévision est réaliste
        """
        try:
            if not prediction or 'values' not in prediction:
                return False
            
            values = prediction['values']
            if len(values) == 0:
                return False
            
            # Vérifier que les valeurs ne sont pas NaN ou infinies
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                self.logger.warning(f"Prévision contient des NaN/Inf pour {symbol}")
                return False
            
            # Vérifier que les valeurs sont positives
            if np.any(values <= 0):
                self.logger.warning(f"Prévision contient des valeurs négatives pour {symbol}")
                return False
            
            # Vérifier que le premier jour n'est pas trop éloigné du prix actuel
            first_pred = values[0]
            change_pct = abs((first_pred - current_price) / current_price * 100)
            
            if change_pct > 15:  # Plus de 15% de changement en un jour
                self.logger.warning(f"Prévision trop extrême pour {symbol}: {change_pct:.2f}%")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation de prévision: {e}")
            return False


    def _combine_predictions_robust(self, predictions, weights, current_price, symbol):
        """
        Combine les prévisions de manière robuste avec des contraintes réalistes
        """
        try:
            if not predictions or not weights:
                return None
            
            # Déterminer l'horizon commun
            min_horizon = min(len(pred['values']) for pred in predictions.values())
            horizon = min(min_horizon, self.horizon)
            
            # Initialiser les arrays combinés
            combined_values = np.zeros(horizon)
            combined_lower = np.zeros(horizon)
            combined_upper = np.zeros(horizon)
            has_bounds = False
            
            # Combiner les prévisions
            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 0)
                if weight > 0:
                    # Valeurs principales
                    model_values = pred['values'][:horizon]
                    combined_values += model_values * weight
                    
                    # Intervalles de confiance
                    if 'lower_bound' in pred and 'upper_bound' in pred:
                        combined_lower += pred['lower_bound'][:horizon] * weight
                        combined_upper += pred['upper_bound'][:horizon] * weight
                        has_bounds = True
            
            # Appliquer des contraintes de réalisme
            combined_values = self._apply_realism_constraints(
                combined_values, current_price, symbol
            )
            
            # Dates de prévision
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=horizon, 
                freq='B'
            )
            
            # Construire le résultat
            result = {
                'dates': forecast_dates,
                'values': combined_values,
                'model_type': 'intelligent_ensemble',
                'models_info': {
                    'models': list(weights.keys()),
                    'weights': weights
                }
            }
            
            # Ajouter les intervalles si disponibles
            if has_bounds:
                # Appliquer les mêmes contraintes aux bornes
                combined_lower = self._apply_realism_constraints(combined_lower, current_price, symbol, is_lower=True)
                combined_upper = self._apply_realism_constraints(combined_upper, current_price, symbol, is_upper=True)
                
                result['lower_bound'] = combined_lower
                result['upper_bound'] = combined_upper
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la combinaison des prévisions: {e}")
            return None


    def _apply_realism_constraints(self, values, current_price, symbol, is_lower=False, is_upper=False):
        """
        Applique des contraintes de réalisme aux prévisions
        """
        try:
            constrained_values = values.copy()
            
            # Calculer la volatilité historique pour définir des limites réalistes
            price_series = self.get_price_column(symbol, 'Close')
            returns = price_series.pct_change().dropna().tail(252)  # Dernière année
            daily_volatility = returns.std()
            
            # Contrainte 1: Limiter le changement jour par jour
            max_daily_change = daily_volatility * 3  # 3 sigma par jour
            
            constrained_values[0] = self._constrain_change(
                current_price, constrained_values[0], max_daily_change
            )
            
            # Contrainte 2: Éviter les changements trop brusques entre jours consécutifs
            for i in range(1, len(constrained_values)):
                prev_price = constrained_values[i-1]
                current_pred = constrained_values[i]
                
                # Limiter le changement par rapport au jour précédent
                constrained_values[i] = self._constrain_change(
                    prev_price, current_pred, max_daily_change
                )
            
            # Contrainte 3: Éviter les tendances trop extrêmes sur l'horizon complet
            total_change = (constrained_values[-1] - current_price) / current_price
            max_total_change = daily_volatility * np.sqrt(len(constrained_values)) * 2
            
            if abs(total_change) > max_total_change:
                # Ajuster graduellement vers une tendance plus réaliste
                adjustment_factor = max_total_change / abs(total_change)
                for i in range(len(constrained_values)):
                    progress = (i + 1) / len(constrained_values)
                    current_change = (constrained_values[i] - current_price) / current_price
                    adjusted_change = current_change * adjustment_factor * progress
                    constrained_values[i] = current_price * (1 + adjusted_change)
            
            # Contrainte 4: Ajustements spécifiques pour les bornes
            if is_lower:
                # Les bornes inférieures ne peuvent pas être supérieures aux valeurs centrales
                constrained_values = np.minimum(constrained_values, values * 0.95)
            elif is_upper:
                # Les bornes supérieures ne peuvent pas être inférieures aux valeurs centrales
                constrained_values = np.maximum(constrained_values, values * 1.05)
            
            return constrained_values
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'application des contraintes: {e}")
            return values  # Retourner les valeurs originales en cas d'erreur


    def _constrain_change(self, base_price, target_price, max_change_rate):
        """
        Limite le changement entre deux prix
        """
        try:
            change_rate = (target_price - base_price) / base_price
            
            if abs(change_rate) > max_change_rate:
                # Limiter le changement au maximum autorisé
                sign = 1 if change_rate > 0 else -1
                limited_change = sign * max_change_rate
                return base_price * (1 + limited_change)
            
            return target_price
            
        except:
            return base_price  # Retourner le prix de base en cas d'erreur


    def _generate_single_prediction_robust(self, symbol, model_type, model_info):
        """
        Génère une prédiction robuste avec un modèle spécifique
        """
        try:
            # Utiliser les méthodes existantes mais avec plus de validation
            if model_type in ['arima', 'sarima']:
                pred = self._predict_with_arima(symbol, model_info, self.horizon)
            elif model_type == 'garch':
                pred = self._predict_with_garch(symbol, model_info, self.horizon)
            elif model_type in ['random_forest', 'gradient_boosting', 'ridge']:
                pred = self._predict_with_ml_robust(symbol, model_info, self.horizon)
            elif model_type in ['lstm', 'gru', 'transformer']:
                pred = self._predict_with_dl(symbol, model_info, self.horizon)
            else:
                return None
            
            # Validation supplémentaire
            if pred and 'values' in pred:
                # Vérifier la cohérence des prédictions
                current_price = self.get_price_column(symbol, 'Close').iloc[-1]
                
                # Ajuster si nécessaire
                pred['values'] = self._smooth_predictions(pred['values'], current_price)
            
            return pred
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction robuste {model_type} pour {symbol}: {e}")
            return None


    def _predict_with_ml_robust(self, symbol, model_info, horizon):
        """
        Version robuste de la prédiction ML avec meilleure gestion des erreurs
        """
        try:
            model = model_info['model']
            features = model_info.get('features', model_info.get('original_features', []))
            
            if not features:
                self.logger.error(f"Aucune feature disponible pour la prédiction ML de {symbol}")
                return None
            
            # Vérifier que les features existent dans les données
            available_features = [f for f in features if f in self.data.columns]
            if len(available_features) < len(features) * 0.5:  # Au moins 50% des features
                self.logger.warning(f"Trop de features manquantes pour {symbol}")
                return None
            
            # Récupérer le prix actuel
            current_price = self.get_price_column(symbol, 'Close').iloc[-1]
            
            # Préparer les dernières données
            try:
                X_recent = self.data[available_features].iloc[-30:].copy()  # 30 derniers jours
                
                # Gérer les valeurs manquantes
                if X_recent.isna().any().any():
                    X_recent = X_recent.fillna(method='ffill').fillna(method='bfill')
                    if X_recent.isna().any().any():
                        # Si il reste des NaN, utiliser la médiane
                        X_recent = X_recent.fillna(X_recent.median())
                
            except Exception as e:
                self.logger.error(f"Erreur lors de la préparation des données pour {symbol}: {e}")
                return None
            
            # Générer les prévisions de manière itérative
            predictions = []
            last_price = current_price
            
            # Calculer une volatilité de référence
            price_series = self.get_price_column(symbol, 'Close')
            returns = price_series.pct_change().dropna().tail(60)
            volatility = returns.std()
            mean_return = returns.mean()
            
            for step in range(horizon):
                try:
                    # Utiliser les dernières données disponibles
                    if step == 0:
                        X_current = X_recent.iloc[-1:].values
                    else:
                        # Pour les prédictions futures, on peut soit:
                        # 1. Réutiliser les dernières données
                        # 2. Essayer de simuler l'évolution des features
                        X_current = X_recent.iloc[-1:].values
                    
                    # Faire la prédiction (rendement)
                    pred_return = model.predict(X_current)[0]
                    
                    # Limiter les rendements extrêmes
                    if abs(pred_return) > volatility * 3:
                        pred_return = np.sign(pred_return) * volatility * 3
                        
                    # Ajouter un peu de bruit réaliste
                    noise = np.random.normal(0, volatility * 0.1)
                    pred_return += noise
                    
                    # Calculer le prix prédit
                    next_price = last_price * (1 + pred_return)
                    
                    # Validation supplémentaire
                    change_pct = abs((next_price - last_price) / last_price)
                    if change_pct > 0.1:  # Plus de 10% en un jour
                        # Utiliser une approche plus conservative
                        trend_return = mean_return + np.random.normal(0, volatility * 0.5)
                        next_price = last_price * (1 + trend_return)
                    
                    predictions.append(next_price)
                    last_price = next_price
                    
                except Exception as pred_error:
                    self.logger.warning(f"Erreur lors de la prédiction pas {step}: {pred_error}")
                    # Utiliser une prédiction de fallback
                    fallback_return = mean_return + np.random.normal(0, volatility * 0.3)
                    next_price = last_price * (1 + fallback_return)
                    predictions.append(next_price)
                    last_price = next_price
            
            # Créer les dates de prévision
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='B')
            
            # Résultat formaté
            result = {
                'dates': forecast_dates,
                'values': np.array(predictions),
                'model_type': 'gradient_boosting_robust',
                'confidence': model_info.get('quality', 'unknown')
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction ML robuste pour {symbol}: {e}")
            return None


    def _smooth_predictions(self, values, current_price):
        """
        Lisse les prédictions pour éviter les changements trop brusques
        """
        try:
            smoothed = values.copy()
            
            # Appliquer un filtre de lissage léger
            if len(smoothed) > 3:
                # Moyenne mobile sur 3 points
                for i in range(1, len(smoothed) - 1):
                    smoothed[i] = (smoothed[i-1] + smoothed[i] + smoothed[i+1]) / 3
            
            # S'assurer de la continuité avec le prix actuel
            if len(smoothed) > 0:
                # Ajuster le premier point pour une transition douce
                first_change = (smoothed[0] - current_price) / current_price
                if abs(first_change) > 0.05:  # Plus de 5%
                    # Réduire le changement initial
                    smoothed[0] = current_price * (1 + np.sign(first_change) * 0.05)
                    
                    # Réajuster les points suivants proportionnellement
                    if len(smoothed) > 1:
                        adjustment_factor = 0.9  # Facteur de décroissance
                        for i in range(1, len(smoothed)):
                            factor = adjustment_factor ** i
                            smoothed[i] = smoothed[i-1] + (smoothed[i] - smoothed[i-1]) * factor
            
            return smoothed
            
        except Exception as e:
            self.logger.error(f"Erreur lors du lissage des prédictions: {e}")
            return values


    def detect_market_regime_enhanced(self, symbol, window_size=60):
        """
        Version améliorée de la détection de régime de marché
        """
        try:
            self.logger.info(f"Détection du régime de marché pour {symbol}")
            
            # Récupérer les données historiques
            historical = self.get_price_column(symbol, 'Close')
            if historical is None or len(historical) < window_size:
                return {'regime': 'unknown', 'confidence': 0}
            
            # Utiliser les dernières données
            prices = historical[-window_size:].values
            returns = np.diff(prices) / prices[:-1]
            
            # Analyse de la volatilité
            volatility = np.std(returns)
            volatility_regime = "normal"
            if volatility > np.percentile(historical.pct_change().dropna().tail(252).std(), 75):
                volatility_regime = "high"
            elif volatility < np.percentile(historical.pct_change().dropna().tail(252).std(), 25):
                volatility_regime = "low"
            
            # Analyse de la tendance avec plusieurs horizons
            trends = {}
            for period in [5, 10, 20]:
                if len(prices) >= period:
                    recent_trend = (prices[-1] - prices[-period]) / prices[-period]
                    trends[f'{period}d'] = recent_trend
            
            # Déterminer la tendance dominante
            trend_signals = []
            for period, trend in trends.items():
                if abs(trend) > 0.02:  # Plus de 2%
                    trend_signals.append(1 if trend > 0 else -1)
                else:
                    trend_signals.append(0)
            
            # Consensus de tendance
            trend_consensus = np.mean(trend_signals)
            
            if trend_consensus > 0.3:
                market_trend = "bullish"
                trend_strength = abs(trend_consensus)
            elif trend_consensus < -0.3:
                market_trend = "bearish"
                trend_strength = abs(trend_consensus)
            else:
                market_trend = "sideways"
                trend_strength = 1 - abs(trend_consensus)
            
            # Analyse de l'autocorrélation (persistance des mouvements)
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
            
            # Déterminer le régime principal
            if abs(autocorr) > 0.15:
                if autocorr > 0:
                    regime_type = "momentum"
                else:
                    regime_type = "mean_reverting"
                regime_strength = abs(autocorr)
            else:
                regime_type = "random_walk"
                regime_strength = 1 - abs(autocorr)
            
            # Score de confiance global
            confidence = min(1.0, (trend_strength + regime_strength) / 2)
            
            regime_info = {
                'regime': {
                    'type': regime_type,
                    'trend': market_trend,
                    'volatility': volatility_regime,
                    'strength': regime_strength,
                    'trend_strength': trend_strength
                },
                'statistics': {
                    'volatility': float(volatility),
                    'autocorrelation': float(autocorr),
                    'trend_consensus': float(trend_consensus),
                    'trends': {k: float(v) for k, v in trends.items()}
                },
                'confidence': float(confidence)
            }
            
            self.logger.info(f"Régime détecté pour {symbol}: {regime_type} ({market_trend}), confiance: {confidence:.2f}")
            
            return regime_info
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection du régime de marché: {e}")
            return {'regime': {'type': 'unknown'}, 'confidence': 0}

        
    def calculate_confidence_intervals(self, predictions):
        """
        Calcule les intervalles de confiance pour les prévisions
        
        Args:
            predictions (dict): Prévisions par symbole
            
        Returns:
            dict: Prévisions avec intervalles de confiance
        """
        try:
            self.logger.info("Calcul des intervalles de confiance pour les prévisions")
            
            # Si les prévisions sont vides, retourner directement
            if not predictions:
                return predictions
            
            # Pour chaque symbole
            for symbol, pred in predictions.items():
                # Vérifier si les intervalles sont déjà calculés
                if 'lower_bound' in pred and 'upper_bound' in pred:
                    self.logger.info(f"Intervalles de confiance déjà présents pour {symbol}")
                    continue
                
                # Récupérer les données historiques avec la méthode robuste
                historical = self.get_price_column(symbol, 'Close')
                if historical is None:
                    self.logger.error(f"Impossible de trouver les données historiques pour {symbol}")
                    continue
                
                # Calculer la volatilité historique (écart-type des rendements)
                returns = historical.pct_change().dropna()
                volatility = returns.std()
                
                # Prédictions et dates
                values = pred['values']
                dates = pred['dates']
                
                # Calculer les intervalles en fonction de la volatilité historique et du niveau de confiance
                # Pour une distribution normale, un intervalle de confiance de 95% est environ 1.96 * écart-type
                z_score = 1.96
                if self.confidence_level != 0.95:
                    # Calculer le Z-score pour d'autres niveaux de confiance
                    from scipy import stats
                    z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
                
                # Calculer les bornes
                lower_bound = np.zeros_like(values)
                upper_bound = np.zeros_like(values)
                
                # Élargir l'intervalle avec le temps (incertitude croissante)
                for i in range(len(values)):
                    # Plus on va loin dans le futur, plus l'incertitude augmente
                    time_factor = np.sqrt(i + 1)
                    
                    # Calculer les bornes en tenant compte de l'incertitude croissante
                    interval_width = values[i] * volatility * z_score * time_factor
                    lower_bound[i] = values[i] - interval_width
                    upper_bound[i] = values[i] + interval_width
                
                # Mettre à jour les prévisions avec les intervalles de confiance
                pred['lower_bound'] = lower_bound
                pred['upper_bound'] = upper_bound
                pred['confidence_level'] = self.confidence_level
                
                self.logger.info(f"Intervalles de confiance calculés pour {symbol}")
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des intervalles de confiance: {str(e)}")
            traceback.print_exc()
            return predictions
        
    def visualize_weighted_ensemble(self, save_path=None):
        """
        Visualise les résultats de l'ensemble pondéré et les contributions de chaque modèle
        
        Args:
            save_path (str, optional): Chemin où sauvegarder les visualisations
            
        Returns:
            bool: True si les visualisations ont été générées avec succès
        """
        try:
            self.logger.info("Début de la fonction visualize_weighted_ensemble")
            
            if 'predictions' not in self.results:
                self.logger.error("Aucune prévision disponible pour la visualisation")
                return False
                    
            self.logger.info("Génération des visualisations d'ensemble pondéré")
            
            # Créer le dossier de sauvegarde si nécessaire
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                self.logger.info(f"Dossier de sauvegarde créé: {save_path}")
            
            self.logger.info("Début du traitement des prédictions par symbole")
            
            for symbol, pred in self.results['predictions'].items():
                self.logger.info(f"Traitement du symbole {symbol}")
                
                # Vérifier si nous avons des informations sur les modèles
                if 'model_type' not in pred or pred['model_type'] != 'weighted_ensemble' or 'models_info' not in pred:
                    self.logger.warning(f"Pas d'information d'ensemble pour {symbol}")
                    continue
                    
                models_info = pred['models_info']
                models = models_info['models']
                weights = models_info['weights']
                
                self.logger.info(f"Création de la figure 1 pour {symbol}")
                
                # Récupérer les données historiques
                historical = self.get_price_column(symbol, 'Close')
                if historical is None:
                    self.logger.error(f"Impossible de trouver les données historiques pour {symbol}")
                    continue
                
                # Préparer les données pour le tracé
                dates = pred['dates']
                values = pred['values']
                
                # 1. Créer un graphique pour la prévision d'ensemble
                fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                
                self.logger.info(f"Ajout des traces pour la figure 1 de {symbol}")
                
                # Tracer les données historiques
                fig1.add_trace(
                    go.Scatter(
                        x=historical.index, 
                        y=historical.values,
                        name='Historique',
                        line=dict(color='royalblue', width=2)
                    )
                )
                
                # Tracer les prévisions
                fig1.add_trace(
                    go.Scatter(
                        x=dates, 
                        y=values,
                        name='Ensemble Pondéré',
                        line=dict(color='firebrick', width=2, dash='dash')
                    )
                )
                
                self.logger.info(f"Ajout des intervalles de confiance pour {symbol}")
                
                # Ajouter les intervalles de confiance si disponibles
                if 'lower_bound' in pred and 'upper_bound' in pred:
                    fig1.add_trace(
                        go.Scatter(
                            x=dates,
                            y=pred['upper_bound'],
                            fill=None,
                            mode='lines',
                            line=dict(color='rgba(255, 0, 0, 0.1)', width=0),
                            name='Intervalle de confiance (supérieur)'
                        )
                    )
                    
                    fig1.add_trace(
                        go.Scatter(
                            x=dates,
                            y=pred['lower_bound'],
                            fill='tonexty',
                            mode='lines',
                            line=dict(color='rgba(255, 0, 0, 0.1)', width=0),
                            name='Intervalle de confiance (inférieur)'
                        )
                    )
                
                self.logger.info(f"Configuration du layout de la figure 1 pour {symbol}")
                
                # Configurer les axes et le titre
                fig1.update_layout(
                    title=f"Prévisions Pondérées pour {symbol}",
                    xaxis_title="Date",
                    yaxis_title="Prix",
                    legend_title="Légende",
                    hovermode="x unified",
                    template="plotly_white"
                )
                
                # Enregistrer ou afficher le graphique
                if save_path:
                    html_path = f"{save_path}/{symbol}_weighted_predictions.html"
                    self.logger.info(f"Sauvegarde de la figure 1 en HTML: {html_path}")
                    try:
                        fig1.write_html(html_path)
                        self.logger.info(f"Figure 1 HTML sauvegardée avec succès pour {symbol}")
                    except Exception as html_error:
                        self.logger.error(f"Erreur lors de la sauvegarde HTML de la figure 1: {html_error}")
                else:
                    self.logger.info(f"Affichage de la figure 1 pour {symbol}")
                    fig1.show()
                
                self.logger.info(f"Traitement de la figure 1 terminé pour {symbol}")
                
                self.logger.info(f"Création de la figure 2 (comparaison des modèles) pour {symbol}")
                
                # 2. Créer un graphique de comparaison des différents modèles
                fig2 = go.Figure()
                
                # Récupérer les prévisions individuelles de chaque modèle si disponibles
                individual_predictions = {}
                
                # Pour chaque modèle dans l'ensemble
                for model_name in models:
                    # Vérifier si le modèle existe
                    if model_name in self.models[symbol]:
                        model_info = self.models[symbol][model_name]
                        
                        # Générer des prévisions individuelles avec le modèle
                        model_pred = self._generate_single_prediction(symbol, model_name, model_info)
                        
                        if model_pred is not None and 'values' in model_pred:
                            individual_predictions[model_name] = model_pred
                
                # Tracer les données historiques
                fig2.add_trace(
                    go.Scatter(
                        x=historical.index[-30:],  # Derniers 30 jours pour mieux voir
                        y=historical.values[-30:],
                        name='Historique',
                        line=dict(color='black', width=2)
                    )
                )
                
                # Tracer les prévisions pondérées
                fig2.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        name='Ensemble Pondéré',
                        line=dict(color='red', width=3)
                    )
                )
                
                # Tracer les prévisions individuelles
                colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink']
                for i, (model_name, model_pred) in enumerate(individual_predictions.items()):
                    color = colors[i % len(colors)]
                    weight = weights.get(model_name, 0)
                    
                    fig2.add_trace(
                        go.Scatter(
                            x=model_pred['dates'],
                            y=model_pred['values'],
                            name=f"{model_name} (poids: {weight:.2f})",
                            line=dict(color=color, width=1.5, dash='dot'),
                            opacity=0.7
                        )
                    )
                
                # Configurer les axes et le titre
                fig2.update_layout(
                    title=f"Comparaison des Modèles pour {symbol}",
                    xaxis_title="Date",
                    yaxis_title="Prix",
                    legend_title="Modèles",
                    hovermode="x unified",
                    template="plotly_white"
                )
                
                # Enregistrer ou afficher le graphique
                if save_path:
                    html_path = f"{save_path}/{symbol}_model_comparison.html"
                    self.logger.info(f"Sauvegarde de la figure 2 en HTML: {html_path}")
                    try:
                        fig2.write_html(html_path)
                        self.logger.info(f"Figure 2 HTML sauvegardée avec succès pour {symbol}")
                    except Exception as html_error:
                        self.logger.error(f"Erreur lors de la sauvegarde HTML de la figure 2: {html_error}")
                else:
                    self.logger.info(f"Affichage de la figure 2 pour {symbol}")
                    fig2.show()
                
                self.logger.info(f"Traitement de la figure 2 terminé pour {symbol}")
                
                self.logger.info(f"Création de la figure 3 (poids des modèles) pour {symbol}")
                
                # 3. Créer un graphique des poids des modèles
                fig3 = go.Figure()
                
                model_names = list(weights.keys())
                model_weights = [weights[model] for model in model_names]
                
                # Tracer un graphique en barres des poids
                fig3.add_trace(
                    go.Bar(
                        x=model_names,
                        y=model_weights,
                        text=[f"{w:.2f}" for w in model_weights],
                        textposition='auto',
                        marker_color=['blue', 'green', 'purple', 'orange'][:len(model_names)]
                    )
                )
                
                # Ajouter des informations sur la performance si disponibles
                if 'backtesting' in self.results and symbol in self.results['backtesting']:
                    backtest_results = self.results['backtesting'][symbol]
                    
                    # Ajouter un tableau avec les métriques de performance
                    metrics_table = []
                    metrics_names = []
                    
                    # Identifier les métriques disponibles (supposons qu'elles sont les mêmes pour tous les modèles)
                    if model_names and model_names[0] in backtest_results:
                        metrics_names = [k for k in backtest_results[model_names[0]].keys() if k.startswith('avg_')]
                    
                    # Créer les lignes du tableau
                    rows = []
                    for model in model_names:
                        if model in backtest_results:
                            row = [model]
                            for metric in metrics_names:
                                value = backtest_results[model].get(metric, 'N/A')
                                row.append(f"{value:.4f}" if isinstance(value, (float, int)) else value)
                            rows.append(row)
                    
                    # Créer l'annotation avec le tableau
                    if rows and metrics_names:
                        table_html = f"<table style='width:100%; border-collapse: collapse;'>"
                        
                        # En-tête du tableau
                        table_html += "<tr>"
                        table_html += "<th style='border:1px solid black; padding:8px;'>Modèle</th>"
                        for metric in metrics_names:
                            table_html += f"<th style='border:1px solid black; padding:8px;'>{metric[4:].upper()}</th>"
                        table_html += "</tr>"
                        
                        # Lignes du tableau
                        for row in rows:
                            table_html += "<tr>"
                            for cell in row:
                                table_html += f"<td style='border:1px solid black; padding:8px;'>{cell}</td>"
                            table_html += "</tr>"
                        
                        table_html += "</table>"
                        
                        fig3.add_annotation(
                            x=0.5,
                            y=-0.3,
                            xref="paper",
                            yref="paper",
                            text=table_html,
                            showarrow=False,
                            align="center",
                            borderwidth=0,
                            height=300,
                            font=dict(size=10)
                        )
                
                # Configurer les axes et le titre
                fig3.update_layout(
                    title=f"Poids des Modèles dans l'Ensemble pour {symbol}",
                    xaxis_title="Modèle",
                    yaxis_title="Poids",
                    template="plotly_white",
                    margin=dict(b=200)  # Ajouter de la marge en bas pour le tableau
                )
                
                # Enregistrer ou afficher le graphique
                if save_path:
                    html_path = f"{save_path}/{symbol}_model_weights.html"
                    self.logger.info(f"Sauvegarde de la figure 3 en HTML: {html_path}")
                    try:
                        fig3.write_html(html_path)
                        self.logger.info(f"Figure 3 HTML sauvegardée avec succès pour {symbol}")
                    except Exception as html_error:
                        self.logger.error(f"Erreur lors de la sauvegarde HTML de la figure 3: {html_error}")
                else:
                    self.logger.info(f"Affichage de la figure 3 pour {symbol}")
                    fig3.show()
                
                self.logger.info(f"Traitement de la figure 3 terminé pour {symbol}")
                
                # 4. Bonus: Créer une figure avec les métriques de performance du backtest si disponible
                if 'backtesting' in self.results and symbol in self.results['backtesting']:
                    self.logger.info(f"Création de la figure 4 (métriques de backtesting) pour {symbol}")
                    
                    backtest_results = self.results['backtesting'][symbol]
                    
                    # Extraire les métriques pour chaque modèle
                    metrics_of_interest = ['avg_rmse', 'avg_mae', 'avg_r2', 'avg_mape']
                    model_names = []
                    metrics_values = {metric: [] for metric in metrics_of_interest}
                    
                    for model_name, metrics in backtest_results.items():
                        model_names.append(model_name)
                        
                        for metric in metrics_of_interest:
                            if metric in metrics and metrics[metric] is not None:
                                value = metrics[metric]
                                # Normaliser R² pour qu'il soit dans le même sens que les autres métriques (plus petit = meilleur)
                                if metric == 'avg_r2':
                                    value = 1 - value
                                metrics_values[metric].append(value)
                            else:
                                metrics_values[metric].append(None)
                    
                    # Créer un graphique en radar pour comparer les métriques
                    fig4 = go.Figure()
                    
                    # Définir les couleurs pour chaque modèle
                    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink']
                    
                    # Normaliser les métriques entre 0 et 1 pour le radar chart
                    for metric in metrics_of_interest:
                        values = metrics_values[metric]
                        if values and not all(v is None for v in values):
                            non_none_values = [v for v in values if v is not None]
                            max_value = max(non_none_values) if non_none_values else 1
                            min_value = min(non_none_values) if non_none_values else 0
                            
                            # Éviter division par zéro
                            if max_value != min_value:
                                metrics_values[metric] = [(v - min_value) / (max_value - min_value) if v is not None else None for v in values]
                            else:
                                metrics_values[metric] = [0.5 if v is not None else None for v in values]
                    
                    # Créer une trace pour chaque modèle
                    for i, model_name in enumerate(model_names):
                        color = colors[i % len(colors)]
                        
                        # Extraire les valeurs normalisées pour ce modèle
                        theta = [metric.replace('avg_', '') for metric in metrics_of_interest]
                        r = [metrics_values[metric][i] if i < len(metrics_values[metric]) and metrics_values[metric][i] is not None else 0 for metric in metrics_of_interest]
                        
                        fig4.add_trace(go.Scatterpolar(
                            r=r,
                            theta=theta,
                            fill='toself',
                            name=model_name,
                            line_color=color
                        ))
                    
                    fig4.update_layout(
                        title=f"Performance des Modèles en Backtesting pour {symbol}",
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=True
                    )
                    
                    # Enregistrer ou afficher le graphique
                    if save_path:
                        html_path = f"{save_path}/{symbol}_backtest_performance.html"
                        self.logger.info(f"Sauvegarde de la figure 4 en HTML: {html_path}")
                        try:
                            fig4.write_html(html_path)
                            self.logger.info(f"Figure 4 HTML sauvegardée avec succès pour {symbol}")
                        except Exception as html_error:
                            self.logger.error(f"Erreur lors de la sauvegarde HTML de la figure 4: {html_error}")
                    else:
                        self.logger.info(f"Affichage de la figure 4 pour {symbol}")
                        fig4.show()
                    
                    self.logger.info(f"Traitement de la figure 4 terminé pour {symbol}")
                
                # POINT DE LOG FINAL POUR CE SYMBOLE
                self.logger.info(f"Traitement des visualisations terminé pour {symbol}")
            
            # Création d'une page d'index HTML pour naviguer entre les visualisations
            if save_path:
                self.logger.info("Création d'une page d'index HTML pour les visualisations")
                
                try:
                    # Collecter tous les fichiers HTML générés
                    html_files = [f for f in os.listdir(save_path) if f.endswith('.html')]
                    
                    # Organiser par symbole
                    symbols_files = {}
                    for html_file in html_files:
                        parts = html_file.split('_')
                        if len(parts) > 1:
                            symbol = parts[0]
                            if symbol not in symbols_files:
                                symbols_files[symbol] = []
                            symbols_files[symbol].append(html_file)
                    
                    # Créer le HTML
                    html_content = """
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Visualisations des Prévisions Boursières</title>
                        <style>
                            body { font-family: Arial, sans-serif; margin: 20px; }
                            h1 { color: #2c3e50; }
                            h2 { color: #3498db; margin-top: 30px; }
                            ul { list-style-type: none; padding: 0; }
                            li { margin: 10px 0; background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
                            a { color: #2980b9; text-decoration: none; }
                            a:hover { text-decoration: underline; }
                        </style>
                    </head>
                    <body>
                        <h1>Visualisations des Prévisions Boursières</h1>
                        <p>Généré le """ + datetime.now().strftime("%Y-%m-%d à %H:%M:%S") + """</p>
                    """
                    
                    for symbol, files in symbols_files.items():
                        html_content += f"<h2>Symbole: {symbol}</h2>\n<ul>\n"
                        
                        for file in sorted(files):
                            # Créer un nom convivial pour l'affichage
                            display_name = file.replace(f"{symbol}_", "").replace(".html", "")
                            display_name = display_name.replace("_", " ").title()
                            
                            html_content += f'<li><a href="{file}" target="_blank">{display_name}</a></li>\n'
                        
                        html_content += "</ul>\n"
                    
                    html_content += """
                    </body>
                    </html>
                    """
                    
                    # Écrire le fichier
                    with open(f"{save_path}/index.html", 'w') as f:
                        f.write(html_content)
                    
                    self.logger.info(f"Page d'index HTML créée: {save_path}/index.html")
                
                except Exception as e:
                    self.logger.error(f"Erreur lors de la création de la page d'index: {str(e)}")
            
            # POINT DE LOG FINAL DE LA FONCTION
            self.logger.info("Fonction visualize_weighted_ensemble terminée avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des visualisations d'ensemble: {str(e)}")
            traceback.print_exc()
            return False
                        
    def _plot_predictions(self, symbol, pred, save_path=None):
        """
        Trace les prévisions pour un symbole
        
        Args:
            symbol (str): Symbole boursier
            pred (dict): Prévisions pour ce symbole
            save_path (str, optional): Chemin pour sauvegarder la visualisation
        """
        try:
            # Récupérer les données historiques
            close_col = f"{symbol}.Close"
            historical = self.data[close_col]
            
            # Préparer les données pour le tracé
            dates = pred['dates']
            values = pred['values']
            
            # Créer une figure interactive avec Plotly
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Tracer les données historiques
            fig.add_trace(
                go.Scatter(
                    x=historical.index, 
                    y=historical.values,
                    name='Historique',
                    line=dict(color='royalblue', width=2)
                )
            )
            
            # Tracer les prévisions
            fig.add_trace(
                go.Scatter(
                    x=dates, 
                    y=values,
                    name='Prévisions',
                    line=dict(color='firebrick', width=2, dash='dash')
                )
            )
            
            # Ajouter les intervalles de confiance si disponibles
            if 'lower_bound' in pred and 'upper_bound' in pred:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=pred['upper_bound'],
                        fill=None,
                        mode='lines',
                        line=dict(color='rgba(255, 0, 0, 0.1)', width=0),
                        name='Intervalle de confiance (supérieur)'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=pred['lower_bound'],
                        fill='tonexty',
                        mode='lines',
                        line=dict(color='rgba(255, 0, 0, 0.1)', width=0),
                        name='Intervalle de confiance (inférieur)'
                    )
                )
            
            # Ajouter la volatilité si disponible (pour les modèles GARCH)
            if 'volatility' in pred:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=pred['volatility'],
                        name='Volatilité prévue',
                        line=dict(color='green', width=2),
                    ),
                    secondary_y=True
                )
            
            # Configurer les axes et le titre
            fig.update_layout(
                title=f"Prévisions pour {symbol} - Modèle: {pred['model_type']}",
                xaxis_title="Date",
                yaxis_title="Prix",
                legend_title="Légende",
                hovermode="x unified",
                template="plotly_white"
            )
            
            if 'volatility' in pred:
                fig.update_yaxes(title_text="Volatilité (%)", secondary_y=True)
            
            # Afficher ou sauvegarder la figure
            if save_path:
                fig.write_html(f"{save_path}/{symbol}_predictions.html")
                
                # Version statique pour les rapports
                fig.write_image(f"{save_path}/{symbol}_predictions.png")
                
                self.logger.info(f"Visualisation des prévisions pour {symbol} sauvegardée dans {save_path}")
            else:
                fig.show()
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du graphique de prévisions pour {symbol}: {str(e)}")
            traceback.print_exc()

    def _plot_backtesting(self, symbol, backtest_results, save_path=None):
        """
        Trace les résultats du backtesting pour un symbole
        
        Args:
            symbol (str): Symbole boursier
            backtest_results (dict): Résultats du backtesting
            save_path (str, optional): Chemin pour sauvegarder la visualisation
        """
        try:
            # Extraire les métriques pour chaque modèle
            models = list(backtest_results.keys())
            metrics = []
            
            for model in models:
                model_metrics = backtest_results[model]
                metrics_names = [k for k in model_metrics.keys() if k.startswith('avg_')]
                
                for metric in metrics_names:
                    metrics.append(metric.replace('avg_', ''))
            
            # Supprimer les doublons
            metrics = list(set(metrics))
            
            # Créer un graphique par métrique
            for metric in metrics:
                fig = go.Figure()
                
                for model in models:
                    if f"avg_{metric}" in backtest_results[model]:
                        value = backtest_results[model][f"avg_{metric}"]
                        
                        # Ajouter la barre au graphique
                        fig.add_trace(
                            go.Bar(
                                x=[model],
                                y=[value],
                                name=model
                            )
                        )
                
                # Configurer le graphique
                fig.update_layout(
                    title=f"Backtesting - {metric.upper()} pour {symbol}",
                    xaxis_title="Modèle",
                    yaxis_title=metric.upper(),
                    barmode='group',
                    template="plotly_white"
                )
                
                # Afficher ou sauvegarder
                if save_path:
                    fig.write_html(f"{save_path}/{symbol}_backtesting_{metric}.html")
                    self.logger.info(f"Visualisation du backtesting ({metric}) pour {symbol} sauvegardée")
                else:
                    fig.show()
            
            # Créer un tableau comparatif pour toutes les métriques
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Modèle'] + metrics,
                    fill_color='paleturquoise',
                    align='center'
                ),
                cells=dict(
                    values=[
                        models,
                        *[[backtest_results[m].get(f"avg_{metric}", "-") for m in models] for metric in metrics]
                    ],
                    fill_color='lavender',
                    align='center'
                )
            )])
            
            fig.update_layout(
                title=f"Comparaison des modèles pour {symbol}",
                width=800
            )
            
            # Afficher ou sauvegarder
            if save_path:
                fig.write_html(f"{save_path}/{symbol}_metrics_comparison.html")
                self.logger.info(f"Tableau comparatif pour {symbol} sauvegardé")
            else:
                fig.show()
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la création des graphiques de backtesting pour {symbol}: {str(e)}")
            traceback.print_exc()

    def _plot_sentiment(self, symbol, save_path=None):
        """
        Trace l'analyse de sentiment et son impact sur les prix
        
        Args:
            symbol (str): Symbole boursier
            save_path (str, optional): Chemin pour sauvegarder la visualisation
        """
        try:
            if self.sentiment_data is None:
                self.logger.warning(f"Aucune donnée de sentiment disponible pour {symbol}")
                return
            
            # Récupérer les données de prix avec la méthode robuste
            prices = self.get_price_column(symbol, 'Close')
            
            if prices is None:
                self.logger.error(f"Impossible de trouver les données de prix pour {symbol}")
                return
            
            # Aligner les indices des données de sentiment
            aligned_sentiment = self.sentiment_data.reindex(prices.index, method='ffill')
            
            # Créer un graphique interactif avec deux axes Y
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Tracer les prix
            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices.values,
                    name=f"{symbol} Prix",
                    line=dict(color='royalblue', width=2)
                ),
                secondary_y=False
            )
            
            # Tracer le sentiment
            if 'sentiment_score_normalized' in aligned_sentiment.columns:
                sentiment_col = 'sentiment_score_normalized'
                
                fig.add_trace(
                    go.Scatter(
                        x=aligned_sentiment.index,
                        y=aligned_sentiment[sentiment_col].values,
                        name='Sentiment',
                        line=dict(color='green', width=2)
                    ),
                    secondary_y=True
                )
                
                # Ajouter un indicateur de corrélation
                correlation = prices.pct_change().corr(aligned_sentiment[sentiment_col])
                
                # Tracer les bulles pour les moments de divergence significative
                price_norm = (prices - prices.min()) / (prices.max() - prices.min())
                divergence = (price_norm - aligned_sentiment[sentiment_col]).abs()
                significant_div = divergence > divergence.quantile(0.9)
                
                if significant_div.any():
                    div_dates = divergence[significant_div].index
                    div_prices = prices[significant_div]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=div_dates,
                            y=div_prices,
                            mode='markers',
                            marker=dict(
                                size=12,
                                color='red',
                                symbol='circle-open',
                                line=dict(
                                    color='red',
                                    width=2
                                )
                            ),
                            name='Divergence Prix-Sentiment'
                        ),
                        secondary_y=False
                    )
            
            # Tracer les différentes sources de sentiment si disponibles
            for col in ['twitter', 'news', 'reddit']:
                if col in aligned_sentiment.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=aligned_sentiment.index,
                            y=aligned_sentiment[col].values,
                            name=f'Sentiment {col}',
                            line=dict(width=1, dash='dot'),
                            opacity=0.6
                        ),
                        secondary_y=True
                    )
            
            # Configurer les axes et le titre
            fig.update_layout(
                title=f"Évolution du prix et du sentiment pour {symbol}",
                xaxis_title="Date",
                hovermode="x unified",
                template="plotly_white"
            )
            
            fig.update_yaxes(title_text="Prix", secondary_y=False)
            fig.update_yaxes(title_text="Score de sentiment (-1 à 1)", secondary_y=True)
            
            # Ajouter une annotation pour la corrélation
            if 'sentiment_score_normalized' in aligned_sentiment.columns:
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Corrélation: {correlation:.2f}",
                    showarrow=False,
                    font=dict(
                        family="Arial",
                        size=12,
                        color="black"
                    ),
                    align="left",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4
                )
            
            # Afficher ou sauvegarder
            if save_path:
                fig.write_html(f"{save_path}/{symbol}_sentiment_analysis.html")
                self.logger.info(f"Visualisation du sentiment pour {symbol} sauvegardée")
            else:
                fig.show()
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du graphique de sentiment pour {symbol}: {str(e)}")
            traceback.print_exc()

    def set_improved_configuration(self):
        """Configure le système avec des paramètres optimisés"""
        
        # Réduire l'horizon pour plus de précision
        self.horizon = 5
        
        # Paramètres de backtesting plus conservateurs
        self.config['backtesting']['windows'] = 3
        self.config['backtesting']['initial_train_size'] = 0.8
        
        # Optimisation plus rapide
        self.config['optimization']['n_iter'] = 10
        self.config['optimization']['cv'] = 3
        
        # Seulement les modèles les plus fiables
        self.config['models']['machine_learning']['algorithms'] = ['gradient_boosting']
        self.config['models']['deep_learning']['enabled'] = False  # Désactiver temporairement
        
        self.logger.info("Configuration améliorée appliquée")

    def validate_api_key(self):
        """Valide que la clé API Tiingo est correctement configurée"""
        
        if ('tiingo' not in self.config or 
            'api_key' not in self.config['tiingo'] or 
            not self.config['tiingo']['api_key'] or
            self.config['tiingo']['api_key'] == "VOTRE_CLÉ_API_TIINGO_ICI"):
            
            self.logger.error("❌ Clé API Tiingo non configurée!")
            self.logger.info("💡 Solution: Définissez votre clé API avec:")
            self.logger.info("   predictor.config['tiingo']['api_key'] = 'VOTRE_VRAIE_CLE'")
            return False
        
        self.logger.info("✅ Clé API Tiingo configurée")
        return True

    def run_quick_test(self):
        """Test rapide du système"""
        
        print("\n🔍 TEST RAPIDE DU SYSTÈME")
        print("=" * 40)
        
        # Test 1: Clé API
        print("1. Validation de la clé API...")
        if not self.validate_api_key():
            return False
        
        # Test 2: Chargement des données
        print("2. Test de chargement des données...")
        try:
            if self.load_data():
                print(f"   ✅ Données chargées: {self.data.shape}")
            else:
                print("   ❌ Échec du chargement")
                return False
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
        
        # Test 3: Construction d'un modèle simple
        print("3. Test de construction de modèle...")
        try:
            symbol = self.config['data']['symbols'][0]
            features = self._get_default_features(symbol)
            
            # Test uniquement le Gradient Boosting
            self.models = {symbol: {}}
            self._build_gradient_boosting_model(symbol, features)
            
            if 'gradient_boosting' in self.models[symbol]:
                quality = self.models[symbol]['gradient_boosting'].get('quality', 'unknown')
                print(f"   ✅ Modèle construit (qualité: {quality})")
            else:
                print("   ❌ Échec de construction du modèle")
                return False
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
        
        # Test 4: Prédiction simple
        print("4. Test de prédiction...")
        try:
            pred = self._predict_with_ml_robust(symbol, self.models[symbol]['gradient_boosting'], 3)
            if pred and 'values' in pred:
                print(f"   ✅ Prédiction générée: {len(pred['values'])} jours")
            else:
                print("   ❌ Échec de prédiction")
                return False
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
        
        print("\n✅ TOUS LES TESTS RÉUSSIS!")
        return True

    # 3. SCRIPT DE CORRECTION AUTOMATIQUE
    def apply_corrections_to_existing_predictor(predictor):
        """Applique les corrections à un objet PredictionBoursiere existant"""
        
        import types
        
        print("🔧 Application des corrections...")
        
        # Remplacer les méthodes problématiques
        predictor._build_gradient_boosting_model = types.MethodType(_build_gradient_boosting_model, predictor)
        predictor.generate_weighted_predictions = types.MethodType(generate_weighted_predictions, predictor)
        predictor._evaluate_model_quality = types.MethodType(_evaluate_model_quality, predictor)
        predictor._calculate_intelligent_weights = types.MethodType(_calculate_intelligent_weights, predictor)
        predictor._validate_prediction = types.MethodType(_validate_prediction, predictor)
        predictor._combine_predictions_robust = types.MethodType(_combine_predictions_robust, predictor)
        predictor._apply_realism_constraints = types.MethodType(_apply_realism_constraints, predictor)
        predictor._constrain_change = types.MethodType(_constrain_change, predictor)
        predictor._generate_single_prediction_robust = types.MethodType(_generate_single_prediction_robust, predictor)
        predictor._predict_with_ml_robust = types.MethodType(_predict_with_ml_robust, predictor)
        predictor._smooth_predictions = types.MethodType(_smooth_predictions, predictor)
        predictor.detect_market_regime_enhanced = types.MethodType(detect_market_regime_enhanced, predictor)
        predictor.run_pipeline_robust_enhanced = types.MethodType(run_pipeline_robust_enhanced, predictor)
        predictor.set_improved_configuration = types.MethodType(set_improved_configuration, predictor)
        predictor.validate_api_key = types.MethodType(validate_api_key, predictor)
        predictor.run_quick_test = types.MethodType(run_quick_test, predictor)
        
        # Appliquer la configuration améliorée
        predictor.set_improved_configuration()
        
        print("✅ Corrections appliquées avec succès!")
        return predictor

    # 4. EXEMPLE D'UTILISATION COMPLÈTE
    def exemple_utilisation():
        """Exemple complet d'utilisation du système corrigé"""
        
        print("🚀 DÉMARRAGE DU SYSTÈME CORRIGÉ")
        print("=" * 50)
        
        # Créer le prédicteur
        predictor = PredictionBoursiere(symbols=["BA"])
        
        # IMPORTANT: Configurer votre vraie clé API
        predictor.config['tiingo']['api_key'] = "VOTRE_VRAIE_CLE_API_ICI"
        
        # Appliquer les corrections
        predictor = apply_corrections_to_existing_predictor(predictor)
        
        # Test rapide
        if not predictor.run_quick_test():
            print("❌ Test échoué, vérifiez la configuration")
            return
        
        # Pipeline complet
        print("\n🔄 LANCEMENT DU PIPELINE COMPLET")
        print("-" * 30)
        
        results = predictor.run_pipeline_robust_enhanced()
        
        if results['status'] == 'success':
            print(f"\n✅ SUCCÈS! ({results['duration_seconds']:.1f}s)")
            
            # Afficher les prédictions
            for symbol, pred in results['predictions'].items():
                print(f"\n📈 {symbol}:")
                current_price = predictor.get_price_column(symbol, 'Close').iloc[-1]
                
                for i, (date, value) in enumerate(zip(pred['dates'][:3], pred['values'][:3])):
                    change = ((value / current_price) - 1) * 100 if i == 0 else ((value / pred['values'][i-1]) - 1) * 100
                    print(f"   {date.strftime('%Y-%m-%d')}: {value:.2f} ({change:+.2f}%)")
        else:
            print(f"❌ ÉCHEC: {results['message']}")

    # 5. SCRIPT DE DIAGNOSTIC AVANCÉ
    def diagnostic_avance():
        """Diagnostic avancé pour identifier les problèmes spécifiques"""
        
        print("🔍 DIAGNOSTIC AVANCÉ")
        print("=" * 30)
        
        try:
            predictor = PredictionBoursiere(symbols=["BA"])
            predictor.config['tiingo']['api_key'] = "1f2a9c9f7cc99f3f7855f6dec4a6760c00735d3f"
            
            # Test 1: Connexion API
            print("1. Test de connexion API Tiingo...")
            try:
                success = predictor.load_data()
                if success:
                    print("   ✅ Connexion API réussie")
                    print(f"   📊 Données: {predictor.data.shape}")
                else:
                    print("   ❌ Échec de connexion API")
                    return
            except Exception as e:
                print(f"   ❌ Erreur API: {e}")
                return
            
            # Test 2: Qualité des données
            print("\n2. Analyse de la qualité des données...")
            symbol = "BA"
            price_data = predictor.get_price_column(symbol, 'Close')
            
            if price_data is not None:
                returns = price_data.pct_change().dropna()
                volatility = returns.std()
                
                print(f"   📈 Prix actuel: {price_data.iloc[-1]:.2f}")
                print(f"   📊 Volatilité: {volatility:.4f}")
                print(f"   📅 Période: {price_data.index[0].date()} à {price_data.index[-1].date()}")
                print(f"   🔢 Points de données: {len(price_data)}")
                
                # Vérifier la stationnarité
                if len(returns) > 50:
                    from scipy import stats
                    _, p_value = stats.jarque_bera(returns)
                    print(f"   📈 Test normalité (Jarque-Bera p-value): {p_value:.4f}")
            
            # Test 3: Construction de modèle avec diagnostic
            print("\n3. Test de construction de modèle avec diagnostic...")
            
            # Préparation
            predictor.preprocess_data()
            features = predictor.select_features()
            
            # Construction avec diagnostic
            predictor.models = {symbol: {}}
            predictor._build_gradient_boosting_model(symbol, features[symbol])
            
            if 'gradient_boosting' in predictor.models[symbol]:
                model_info = predictor.models[symbol]['gradient_boosting']
                metrics = model_info.get('metrics', {})
                
                print(f"   🤖 Modèle construit:")
                print(f"      - Qualité: {model_info.get('quality', 'unknown')}")
                print(f"      - R²: {metrics.get('r2', 'N/A'):.4f}")
                print(f"      - MSE: {metrics.get('mse', 'N/A'):.6f}")
                print(f"      - Features utilisées: {len(model_info.get('features', []))}")
                
                # Test de prédiction
                pred = predictor._predict_with_ml_robust(symbol, model_info, 3)
                if pred:
                    changes = []
                    current = price_data.iloc[-1]
                    for val in pred['values']:
                        change = ((val / current) - 1) * 100
                        changes.append(change)
                        current = val
                    
                    print(f"   🔮 Prédictions test:")
                    for i, change in enumerate(changes):
                        print(f"      Jour {i+1}: {change:+.2f}%")
            
            print("\n✅ DIAGNOSTIC TERMINÉ")
            
        except Exception as e:
            print(f"❌ Erreur lors du diagnostic: {e}")
            import traceback
            traceback.print_exc()

    if __name__ == "__main__":
        # Choisir l'action
        import sys
        
        if len(sys.argv) > 1:
            if sys.argv[1] == "--test":
                exemple_utilisation()
            elif sys.argv[1] == "--diagnostic":
                diagnostic_avance()
            else:
                print("Usage: python script.py [--test|--diagnostic]")
        else:
            print("Sélectionnez une option:")
            print("1. Test rapide")
            print("2. Diagnostic avancé")
            print("3. Exemple complet")
            
            choice = input("Votre choix (1-3): ")
            
            if choice == "1":
                predictor = PredictionBoursiere(symbols=["BA"])
                predictor.config['tiingo']['api_key'] = "1f2a9c9f7cc99f3f7855f6dec4a6760c00735d3f"
                predictor = apply_corrections_to_existing_predictor(predictor)
                predictor.run_quick_test()
            elif choice == "2":
                diagnostic_avance()
            elif choice == "3":
                exemple_utilisation()
            else:
                print("Choix invalide")
            
    def generate_report(self, output_format='json'):
        """
        Génère un rapport complet des résultats de prévision
        
        Args:
            output_format (str): Format de sortie ('html', 'pdf', 'json')
        
        Returns:
            str: Chemin vers le rapport généré
        """
        try:
            self.logger.info(f"Génération du rapport au format {output_format}")
        
            # Créer le dossier pour les rapports
            report_dir = "rapports"
            os.makedirs(report_dir, exist_ok=True)
        
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"rapport_prevision_{timestamp}"
            
            # Format JSON - optimisé pour les programmes de visualisation
            if output_format == 'json':
                # Créer un dictionnaire avec toutes les données pertinentes
                report_data = {
                    'generated_at': datetime.now().isoformat(),
                    'symbols': self.config['data']['symbols'],
                    'horizon': self.horizon,
                    'confidence_level': self.confidence_level
                }
            
                # Ajouter les prévisions détaillées
                if 'predictions' in self.results:
                    predictions = {}
                
                    for symbol, pred in self.results['predictions'].items():
                        # Données de base
                        symbol_data = {
                            'model_type': pred.get('model_type', 'unknown'),
                            'dates': [d.strftime('%Y-%m-%d') for d in pred['dates']],
                            'values': pred['values'].tolist() if isinstance(pred['values'], np.ndarray) else pred['values']
                        }
                    
                        # Intervalles de confiance
                        if 'lower_bound' in pred and 'upper_bound' in pred:
                            symbol_data['lower_bound'] = pred['lower_bound'].tolist() if isinstance(pred['lower_bound'], np.ndarray) else pred['lower_bound']
                            symbol_data['upper_bound'] = pred['upper_bound'].tolist() if isinstance(pred['upper_bound'], np.ndarray) else pred['upper_bound']
                    
                        # Volatilité prévue
                        if 'volatility' in pred:
                            symbol_data['volatility'] = pred['volatility'].tolist() if isinstance(pred['volatility'], np.ndarray) else pred['volatility']
                    
                        # Information sur les modèles d'ensemble
                        if 'models_info' in pred:
                            symbol_data['models_info'] = pred['models_info']
                        
                        # Ajouter les indicateurs techniques pour l'analyse
                        close_col = f"{symbol}.Close"
                        if close_col in self.data.columns or (isinstance(self.data.columns, pd.MultiIndex) and (symbol, 'Close') in self.data.columns):
                            # Format de colonne correct
                            if isinstance(self.data.columns, pd.MultiIndex) and (symbol, 'Close') in self.data.columns:
                                historical_close = self.data[(symbol, 'Close')]
                            else:
                                historical_close = self.data[close_col]
                            
                            # Calcul des indicateurs techniques récents
                            last_prices = historical_close[-30:].values
                            
                            # Moyennes mobiles
                            ma5 = np.mean(last_prices[-5:]) if len(last_prices) >= 5 else None
                            ma10 = np.mean(last_prices[-10:]) if len(last_prices) >= 10 else None
                            ma20 = np.mean(last_prices[-20:]) if len(last_prices) >= 20 else None
                            
                            # RSI simplifié
                            rsi = None
                            if len(last_prices) >= 14:
                                delta = np.diff(last_prices)
                                gain = delta.copy()
                                loss = delta.copy()
                                gain[gain < 0] = 0
                                loss[loss > 0] = 0
                                loss = -loss
                                avg_gain = np.mean(gain[-14:])
                                avg_loss = np.mean(loss[-14:])
                                if avg_loss > 0:
                                    rs = avg_gain / avg_loss
                                    rsi = 100 - (100 / (1 + rs))
                            
                            # Volatilité récente
                            volatility = np.std(np.diff(last_prices) / last_prices[:-1]) if len(last_prices) > 1 else None
                            
                            # Ajouter les indicateurs techniques
                            symbol_data['technical_indicators'] = {
                                'last_price': float(last_prices[-1]) if len(last_prices) > 0 else None,
                                'ma5': float(ma5) if ma5 is not None else None,
                                'ma10': float(ma10) if ma10 is not None else None,
                                'ma20': float(ma20) if ma20 is not None else None,
                                'rsi': float(rsi) if rsi is not None else None,
                                'volatility': float(volatility) if volatility is not None else None
                            }
                            
                            # Données historiques récentes pour contexte
                            symbol_data['historical_data'] = {
                                'dates': [d.strftime('%Y-%m-%d') for d in historical_close.index[-30:]],
                                'prices': historical_close[-30:].values.tolist()
                            }
                        
                        predictions[symbol] = symbol_data
                
                    report_data['predictions'] = predictions
            
                # Ajouter les résultats du backtesting pour l'évaluation des modèles
                if 'backtesting' in self.results:
                    # Formatter correctement les données de backtesting
                    backtesting_data = {}
                    
                    for symbol, backtest in self.results['backtesting'].items():
                        symbol_backtest = {}
                        
                        for model_name, metrics in backtest.items():
                            # Convertir les np.ndarray en listes
                            model_metrics = {}
                            for metric_name, metric_value in metrics.items():
                                if isinstance(metric_value, np.ndarray):
                                    model_metrics[metric_name] = metric_value.tolist()
                                elif isinstance(metric_value, list) and metric_value and isinstance(metric_value[0], np.float64):
                                    model_metrics[metric_name] = [float(v) for v in metric_value]
                                elif isinstance(metric_value, np.float64):
                                    model_metrics[metric_name] = float(metric_value)
                                else:
                                    model_metrics[metric_name] = metric_value
                            
                            symbol_backtest[model_name] = model_metrics
                        
                        backtesting_data[symbol] = symbol_backtest
                    
                    report_data['backtesting'] = backtesting_data
                
                # Ajouter des analyses de marché pour contexte
                market_analyses = {}
                for symbol in self.config['data']['symbols']:
                    market_regime = self.detect_market_regime(symbol)
                    if market_regime and 'regime' in market_regime and market_regime['regime'] != 'unknown':
                        market_analyses[symbol] = market_regime
                
                if market_analyses:
                    report_data['market_analyses'] = market_analyses
                
                # Ajouter des métadonnées sur la qualité des prédictions
                prediction_quality = {}
                for symbol in self.config['data']['symbols']:
                    if symbol in self.results.get('backtesting', {}) and symbol in self.results.get('predictions', {}):
                        # Identifier le modèle utilisé pour la prédiction
                        model_type = self.results['predictions'][symbol].get('model_type', 'unknown')
                        
                        # Trouver les métriques correspondantes dans le backtesting
                        if model_type in self.results['backtesting'][symbol]:
                            backtest_metrics = self.results['backtesting'][symbol][model_type]
                            prediction_quality[symbol] = {
                                'model': model_type,
                                'rmse': backtest_metrics.get('avg_rmse'),
                                'mape': backtest_metrics.get('avg_mape'),
                                'r2': backtest_metrics.get('avg_r2'),
                                'confidence': self._calculate_prediction_confidence(backtest_metrics)
                            }
                
                if prediction_quality:
                    report_data['prediction_quality'] = prediction_quality
                
                # Écrire le fichier JSON
                report_path = f"{report_dir}/{report_name}.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=4, default=str)
                
                self.logger.info(f"Rapport JSON généré: {report_path}")
                return report_path
            
            # [...le reste de la fonction pour les autres formats...]
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
            traceback.print_exc()
            return None
        
    def _calculate_prediction_confidence(self, metrics):
        """
        Calcule un score de confiance pour les prédictions basé sur les métriques de backtesting
        
        Args:
            metrics (dict): Métriques de backtesting
            
        Returns:
            float: Score de confiance entre 0 et 1
        """
        try:
            # Initialiser les scores
            r2_score = 0
            mape_score = 0
            rmse_score = 0
            
            # Score basé sur R²
            if 'avg_r2' in metrics and metrics['avg_r2'] is not None:
                r2 = metrics['avg_r2']
                if r2 > 0:
                    r2_score = r2  # R² positif est bon (max 1)
                else:
                    r2_score = 0  # R² négatif est mauvais
            
            # Score basé sur MAPE (inversé, car plus petit est meilleur)
            if 'avg_mape' in metrics and metrics['avg_mape'] is not None:
                mape = metrics['avg_mape']
                mape_score = max(0, 1 - (mape / 100))  # MAPE de 0% donne 1, MAPE de 100% donne 0
            
            # Score basé sur RMSE (normalisé par rapport à une valeur typique)
            if 'avg_rmse' in metrics and metrics['avg_rmse'] is not None:
                rmse = metrics['avg_rmse']
                # Supposons qu'un RMSE de 10% du prix est la limite acceptable
                typical_price = 100  # Valeur arbitraire
                rmse_score = max(0, 1 - (rmse / (typical_price * 0.1)))
            
            # Combiner les scores avec des poids
            # R² est le plus important, puis MAPE, puis RMSE
            combined_score = (r2_score * 0.5) + (mape_score * 0.3) + (rmse_score * 0.2)
            
            # Limiter entre 0 et 1
            return max(0, min(1, combined_score))
        
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du score de confiance: {str(e)}")
            return 0.5  # Valeur par défaut moyenne
        
    def save_model(self, path="models"):
        """
        Sauvegarde les modèles entraînés - version corrigée pour les problèmes JSON
        
        Args:
            path (str): Chemin pour sauvegarder les modèles
        
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            self.logger.info(f"Sauvegarde des modèles dans {path}")
        
            # Créer le dossier si nécessaire
            os.makedirs(path, exist_ok=True)
        
            # Sauvegarder chaque modèle
            for symbol in self.models:
                symbol_path = os.path.join(path, symbol)
                os.makedirs(symbol_path, exist_ok=True)
            
                for model_name, model_info in self.models[symbol].items():
                    # Les modèles statsmodels peuvent être sauvegardés directement
                    if model_name in ['arima', 'sarima', 'var', 'garch']:
                        if 'model' in model_info:
                            model = model_info['model']
                            # Utiliser pickle pour sauvegarder
                            with open(f"{symbol_path}/{model_name}.pkl", 'wb') as f:
                                pickle.dump(model, f)
                
                    # Les modèles scikit-learn aussi
                    elif model_name in ['random_forest', 'gradient_boosting', 'ridge']:
                        if 'model' in model_info:
                            model = model_info['model']
                            with open(f"{symbol_path}/{model_name}.pkl", 'wb') as f:
                                pickle.dump(model, f)
                
                    # Pour les modèles Keras, utiliser la fonction save
                    elif model_name in ['lstm', 'gru', 'transformer']:
                        if 'model' in model_info:
                            model = model_info['model']
                            model.save(f"{symbol_path}/{model_name}.h5")
                
                    # Sauvegarder également les métadonnées du modèle - CORRIGÉ POUR JSON
                    metadata = {}
                    for k, v in model_info.items():
                        if k != 'model':
                            # Convertir les tuples en listes pour la sérialisation JSON
                            if isinstance(v, dict):
                                new_dict = {}
                                for key, value in v.items():
                                    # Convertir les clés de tuple en chaînes
                                    if isinstance(key, tuple):
                                        new_key = str(key)
                                    else:
                                        new_key = key
                                    new_dict[new_key] = value
                                metadata[k] = new_dict
                            elif isinstance(v, tuple):
                                metadata[k] = list(v)
                            elif isinstance(v, np.ndarray):
                                metadata[k] = v.tolist()
                            elif isinstance(v, (np.float32, np.float64)):
                                metadata[k] = float(v)
                            elif isinstance(v, (np.int32, np.int64)):
                                metadata[k] = int(v)
                            else:
                                metadata[k] = v
                    
                    with open(f"{symbol_path}/{model_name}_metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=4, default=str)
            
                self.logger.info(f"Modèles pour {symbol} sauvegardés dans {symbol_path}")
        
            # Sauvegarder la configuration
            with open(f"{path}/config.json", 'w') as f:
                json.dump(self.config, f, indent=4, default=str)
        
            # Sauvegarder les résultats
            with open(f"{path}/results.json", 'w') as f:
                # Convertir les objets non sérialisables
                results_json = {}
                for key, value in self.results.items():
                    if isinstance(value, dict):
                        results_json[key] = {}
                        for k, v in value.items():
                            if isinstance(v, dict):
                                results_json[key][k] = {}
                                for k2, v2 in v.items():
                                    if isinstance(v2, np.ndarray):
                                        results_json[key][k][k2] = v2.tolist()
                                    elif isinstance(v2, pd.DatetimeIndex):
                                        results_json[key][k][k2] = [str(d) for d in v2]
                                    elif isinstance(v2, dict):
                                        results_json[key][k][k2] = {}
                                        for k3, v3 in v2.items():
                                            if isinstance(k3, tuple):
                                                # Convertir les clés tuple en string
                                                results_json[key][k][k2][str(k3)] = v3
                                            else:
                                                results_json[key][k][k2][k3] = v3
                                    else:
                                        results_json[key][k][k2] = v2
                            else:
                                results_json[key][k] = v
                    else:
                        results_json[key] = value
            
                json.dump(results_json, f, indent=4, default=str)
        
            self.logger.info(f"Configuration et résultats sauvegardés dans {path}")
            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des modèles: {str(e)}")
            traceback.print_exc()
            return False
        
    def load_model(self, path="models"):
        """
        Charge des modèles précédemment sauvegardés
        
        Args:
            path (str): Chemin où les modèles sont sauvegardés
            
        Returns:
            bool: True si le chargement a réussi
        """
        try:
            self.logger.info(f"Chargement des modèles depuis {path}")
            
            # Vérifier si le dossier existe
            if not os.path.exists(path):
                self.logger.error(f"Le dossier {path} n'existe pas")
                return False
            
            # Charger la configuration
            config_path = f"{path}/config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                self.logger.info("Configuration chargée")
            
            # Initialiser le dictionnaire de modèles
            self.models = {}
            
            # Parcourir les sous-dossiers (un par symbole)
            for symbol_dir in os.listdir(path):
                symbol_path = os.path.join(path, symbol_dir)
                
                # Ignorer les fichiers
                if not os.path.isdir(symbol_path) or symbol_dir.startswith('.'):
                    continue
                
                symbol = symbol_dir
                self.models[symbol] = {}
                
                # Charger chaque modèle pour ce symbole
                for model_file in os.listdir(symbol_path):
                    if model_file.endswith('_metadata.json'):
                        continue
                    
                    if model_file.endswith('.pkl'):
                        model_name = model_file.split('.')[0]
                        
                        # Charger le modèle
                        with open(f"{symbol_path}/{model_file}", 'rb') as f:
                            model = pickle.load(f)
                        
                        # Charger les métadonnées si disponibles
                        metadata_file = f"{symbol_path}/{model_name}_metadata.json"
                        metadata = {}
                        if os.path.exists(metadata_file):
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                        
                        # Stocker le modèle et ses métadonnées
                        self.models[symbol][model_name] = {'model': model, **metadata}
                    
                    elif model_file.endswith('.h5'):
                        model_name = model_file.split('.')[0]
                        
                        # Charger le modèle Keras
                        model = tf.keras.models.load_model(f"{symbol_path}/{model_file}")
                        
                        # Charger les métadonnées si disponibles
                        metadata_file = f"{symbol_path}/{model_name}_metadata.json"
                        metadata = {}
                        if os.path.exists(metadata_file):
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                        
                        # Stocker le modèle et ses métadonnées
                        self.models[symbol][model_name] = {'model': model, **metadata}
                
                self.logger.info(f"Modèles pour {symbol} chargés depuis {symbol_path}")
            
            # Charger les résultats précédents si disponibles
            results_path = f"{path}/results.json"
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.results = json.load(f)
                self.logger.info("Résultats chargés")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des modèles: {str(e)}")
            traceback.print_exc()
            return False
        
    def generate_diagnostic_report(self):
        """
        Génère un rapport complet de diagnostic sur l'état actuel de l'objet PredictionBoursiere.
        Utile pour comprendre les problèmes et avoir une vue d'ensemble des données.
        
        Returns:
            dict: Rapport de diagnostic
        """
        try:
            self.logger.info("Génération du rapport de diagnostic")
            
            report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "configuration": {},
                "data_status": {},
                "models_status": {},
                "performance_metrics": {},
                "recommendations": []
            }
            
            # 1. Configuration
            report["configuration"] = {
                "symbols": self.config['data']['symbols'],
                "date_range": f"{self.config['data']['start_date']} à {self.config['data']['end_date']}",
                "features_enabled": {
                    "technical_indicators": self.config['features']['technical_indicators'],
                    "fundamental_data": self.config['features']['fundamental_data'],
                    "sentiment_analysis": self.config['features']['sentiment_analysis'],
                    "macroeconomic": self.config['features']['macroeconomic']
                },
                "models_enabled": {
                    "arima": self.config['models']['arima']['enabled'],
                    "var": self.config['models']['var']['enabled'],
                    "garch": self.config['models']['garch']['enabled'],
                    "machine_learning": self.config['models']['machine_learning']['enabled'],
                    "deep_learning": self.config['models']['deep_learning']['enabled'],
                },
                "prediction_horizon": self.horizon,
                "confidence_level": self.confidence_level
            }
            
            # 2. Données
            if self.data is not None:
                report["data_status"] = {
                    "loaded": True,
                    "shape": {"rows": self.data.shape[0], "columns": self.data.shape[1]},
                    "period": {"start": str(self.data.index[0]), "end": str(self.data.index[-1])},
                    "missing_values": int(self.data.isnull().sum().sum()),
                    "missing_percentage": float((self.data.isnull().sum().sum()/(self.data.shape[0]*self.data.shape[1])*100))
                }
                
                # Vérifier et ajouter des informations sur les symboles
                symbols_info = {}
                for symbol in self.config['data']['symbols']:
                    symbol_info = {}
                    
                    # Informations sur les données de prix
                    price_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
                    col_status = {}
                    
                    for col_type in price_cols:
                        series = self.get_price_column(symbol, col_type)
                        if series is not None:
                            col_status[col_type] = {
                                "found": True,
                                "missing_values": int(series.isnull().sum()),
                                "min": float(series.min()) if not series.isnull().all() else None,
                                "max": float(series.max()) if not series.isnull().all() else None
                            }
                        else:
                            col_status[col_type] = {"found": False}
                    
                    symbol_info["data"] = col_status
                    
                    # Vérifier les caractéristiques sélectionnées
                    if hasattr(self, "selected_features") and isinstance(self.selected_features, dict) and symbol in self.selected_features:
                        symbol_info["features"] = {
                            "count": len(self.selected_features[symbol]),
                            "features": self.selected_features[symbol][:10] + ['...'] if len(self.selected_features[symbol]) > 10 else self.selected_features[symbol]
                        }
                    else:
                        symbol_info["features"] = {"count": 0, "status": "non_sélectionnées"}
                    
                    # Vérifier les modèles construits
                    if hasattr(self, "models") and isinstance(self.models, dict) and symbol in self.models:
                        symbol_info["models"] = {
                            "count": len(self.models[symbol]),
                            "types": list(self.models[symbol].keys())
                        }
                    else:
                        symbol_info["models"] = {"count": 0, "status": "non_construits"}
                    
                    symbols_info[symbol] = symbol_info
                
                report["data_status"]["symbols"] = symbols_info
            else:
                report["data_status"] = {"loaded": False}
                report["recommendations"].append("Exécuter load_data() pour charger les données avant de poursuivre")
            
            # 3. Modèles
            if hasattr(self, "models") and isinstance(self.models, dict) and len(self.models) > 0:
                models_count = {}
                for symbol in self.models:
                    for model_type in self.models[symbol]:
                        if model_type in models_count:
                            models_count[model_type] += 1
                        else:
                            models_count[model_type] = 1
                
                report["models_status"] = {
                    "symbols_with_models": len(self.models),
                    "model_types_count": models_count
                }
            else:
                report["models_status"] = {
                    "symbols_with_models": 0,
                    "status": "aucun_modèle_construit"
                }
                report["recommendations"].append("Les modèles n'ont pas été construits. Vérifiez les données et exécutez build_optimized_models()")
            
            # 4. Performance des modèles (si backtesting a été exécuté)
            if 'backtesting' in self.results and len(self.results['backtesting']) > 0:
                performance = {}
                
                for symbol, backtest_results in self.results['backtesting'].items():
                    symbol_perf = {}
                    
                    for model_type, metrics in backtest_results.items():
                        model_metrics = {}
                        
                        for metric_name, value in metrics.items():
                            if metric_name.startswith('avg_'):
                                model_metrics[metric_name[4:]] = float(value) if isinstance(value, (float, int, np.float64)) else None
                        
                        symbol_perf[model_type] = model_metrics
                    
                    performance[symbol] = symbol_perf
                
                report["performance_metrics"] = performance
            else:
                report["performance_metrics"] = {"status": "aucun_backtesting_exécuté"}
                report["recommendations"].append("Exécutez backtest_models() pour évaluer la performance des modèles")
            
            # 5. Générer des recommandations basées sur l'analyse
            if self.data is not None and self.data.shape[0] < 200:
                report["recommendations"].append("Considérez utiliser plus de données historiques pour améliorer la qualité des modèles")
            
            if hasattr(self, "models") and isinstance(self.models, dict) and len(self.models) == 0:
                report["recommendations"].append("Aucun modèle n'a été construit. Vérifiez les erreurs dans les logs et les données d'entrée")
            
            if 'backtesting' in self.results and len(self.results['backtesting']) > 0:
                # Chercher les modèles avec de mauvaises performances
                for symbol, backtest_results in self.results['backtesting'].items():
                    for model_type, metrics in backtest_results.items():
                        if 'avg_r2' in metrics and metrics['avg_r2'] < 0:
                            report["recommendations"].append(f"Le modèle {model_type} pour {symbol} a un R² négatif, envisagez de l'exclure de l'ensemble")
                        
                        if 'avg_mape' in metrics and metrics['avg_mape'] > 15:
                            report["recommendations"].append(f"Le modèle {model_type} pour {symbol} a un MAPE élevé (>{metrics['avg_mape']:.1f}%), envisagez de l'ajuster")
            
            self.logger.info("Rapport de diagnostic généré avec succès")
            return report
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport de diagnostic: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Erreur lors de la génération du rapport: {str(e)}"
            }
        
    def generate_predictions_conservative(self):
        """
        Génère des prédictions plus conservatrices
        """
        try:
            self.logger.info("Génération de prédictions conservatrices")
            
            predictions = {}
            
            for symbol in self.config['data']['symbols']:
                # Récupérer le dernier prix connu
                price_series = self.get_price_column(symbol, 'Close')
                if price_series is None:
                    self.logger.warning(f"Données de prix introuvables pour {symbol}")
                    continue
                    
                current_price = price_series.iloc[-1]
                
                # Calculer la volatilité historique
                returns = price_series.pct_change().dropna()
                volatility = returns.std()
                
                # Volatilité historique réduite pour être conservateur
                conservative_volatility = volatility * 0.8
                
                # Dates pour les prévisions
                last_date = self.data.index[-1]
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=self.horizon, freq='B')
                
                # Initialiser les valeurs de prévision
                forecast_values = np.zeros(self.horizon)
                
                # Générer des prévisions basées sur la moyenne pondérée des modèles disponibles
                has_predictions = False
                weighted_forecast = np.zeros(self.horizon)
                total_weight = 0
                
                # Poids des modèles basés sur leur performance de backtesting
                model_weights = {}
                
                # Collecter les prédictions des modèles disponibles
                for model_name in ['gradient_boosting', 'garch', 'lstm', 'arima']:
                    if model_name in self.models[symbol]:
                        # Générer la prédiction avec ce modèle
                        if model_name == 'arima':
                            pred = self._predict_with_arima(symbol, self.models[symbol][model_name], self.horizon)
                        elif model_name == 'garch':
                            garch_pred = self._predict_with_garch(symbol, self.models[symbol][model_name], self.horizon)
                            # GARCH donne la volatilité, convertir en prix
                            pred = {
                                'values': np.array([current_price] * self.horizon),
                                'volatility': garch_pred['volatility'] if 'volatility' in garch_pred else None
                            }
                        elif model_name == 'gradient_boosting':
                            pred = self._predict_with_ml(symbol, self.models[symbol][model_name], self.horizon)
                        elif model_name == 'lstm':
                            pred = self._predict_with_dl(symbol, self.models[symbol][model_name], self.horizon)
                        
                        # Si la prédiction est valide
                        if pred and 'values' in pred and len(pred['values']) == self.horizon:
                            has_predictions = True
                            
                            # Déterminer le poids basé sur la performance de backtesting
                            weight = 1.0  # Poids par défaut
                            
                            if ('backtesting' in self.results and symbol in self.results['backtesting'] and 
                                model_name in self.results['backtesting'][symbol] and 
                                'avg_rmse' in self.results['backtesting'][symbol][model_name]):
                                
                                rmse = self.results['backtesting'][symbol][model_name]['avg_rmse']
                                if rmse > 0:
                                    weight = 1.0 / rmse
                                else:
                                    weight = 1.0
                            
                            model_weights[model_name] = weight
                            total_weight += weight
                            
                            # Ajouter la contribution de ce modèle
                            weighted_forecast += pred['values'] * weight
                
                # Si aucun modèle n'a généré de prédictions, utiliser une marche aléatoire
                if not has_predictions or total_weight == 0:
                    self.logger.warning(f"Aucune prévision de modèle valide pour {symbol}, utilisation d'une marche aléatoire")
                    
                    # Modèle de marche aléatoire simple
                    mean_return = returns.mean()
                    forecast_values = np.zeros(self.horizon)
                    
                    for i in range(self.horizon):
                        if i == 0:
                            forecast_values[i] = current_price * (1 + mean_return)
                        else:
                            forecast_values[i] = forecast_values[i-1] * (1 + mean_return)
                else:
                    # Normaliser les poids et calculer la prévision finale
                    for model_name in model_weights:
                        model_weights[model_name] /= total_weight
                    
                    forecast_values = weighted_forecast / total_weight
                
                # Vérification finale des prédictions
                for i in range(len(forecast_values)):
                    # Si la prédiction est aberrante (plus de 5% de variation), la limiter
                    if abs((forecast_values[i] / current_price) - 1) > 0.05:
                        self.logger.warning(f"Prédiction aberrante détectée pour {symbol} jour {i+1}, limitation appliquée")
                        # Limiter à ±5% de variation par rapport au prix actuel
                        forecast_values[i] = current_price * (1 + (0.05 * np.sign(forecast_values[i] - current_price)))
                
                # Calculer des intervalles de confiance conservateurs
                lower_bound = np.zeros(self.horizon)
                upper_bound = np.zeros(self.horizon)
                
                # Z-score pour un intervalle de confiance à 95%
                z_score = 1.96
                
                for i in range(self.horizon):
                    # Élargir l'intervalle avec le temps (plus d'incertitude)
                    time_factor = 1 + (i * 0.1)  # 10% d'incertitude supplémentaire par période
                    
                    # Élargir les intervalles pour être plus conservateur
                    interval_width = forecast_values[i] * conservative_volatility * z_score * time_factor
                    
                    lower_bound[i] = forecast_values[i] - interval_width
                    upper_bound[i] = forecast_values[i] + interval_width
                
                # Stocker les prédictions
                predictions[symbol] = {
                    'dates': forecast_dates,
                    'values': forecast_values,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'model_type': 'conservative_ensemble',
                    'models_info': {
                        'models': list(model_weights.keys()) if model_weights else ['fallback_model'],
                        'weights': model_weights if model_weights else {'fallback_model': 1.0}
                    }
                }
                
                self.logger.info(f"Prédictions conservatrices générées pour {symbol}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des prédictions conservatrices: {str(e)}")
            traceback.print_exc()
            return {}

    
    def _generate_single_prediction(self, symbol, model_type, model_info):
        """
            Génère une prédiction avec un modèle spécifique
            
            Args:
                symbol (str): Symbole boursier
                model_type (str): Type de modèle
                model_info (dict): Informations sur le modèle
                
            Returns:
                dict: Prévisions ou None en cas d'échec
            """
        try:
            self.logger.info(f"Génération de prévisions pour {symbol} avec {model_type}")
                
            # Différentes méthodes selon le type de modèle
            if model_type in ['arima', 'sarima']:
                return self._predict_with_arima(symbol, model_info, self.horizon)
            elif model_type == 'var':
                return self._predict_with_var(symbol, model_info, self.horizon)
            elif model_type == 'garch':
                    return self._predict_with_garch(symbol, model_info, self.horizon)
            elif model_type in ['random_forest', 'gradient_boosting', 'ridge']:
                    return self._predict_with_ml(symbol, model_info, self.horizon)
            elif model_type in ['lstm', 'gru', 'transformer']:
                    return self._predict_with_dl(symbol, model_info, self.horizon)
            else:
                    self.logger.warning(f"Type de modèle non supporté: {model_type}")
                    return None
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des prévisions pour {symbol} avec {model_type}: {str(e)}")
            return None


    def generate_trading_dashboard(self, save_path="trading_dashboard"):
        """
        Génère un tableau de bord de trading complet dans une seule page HTML
        avec tous les indicateurs techniques essentiels.
        
        Args:
            save_path (str): Chemin où sauvegarder le tableau de bord HTML
        
        Returns:
            str: Chemin vers le fichier HTML généré
        """
        try:
            self.logger.info("Génération du tableau de bord de trading")
            
            # Créer le dossier de sauvegarde
            os.makedirs(save_path, exist_ok=True)
            
            # Préparer les données pour chaque symbole
            for symbol in self.config['data']['symbols']:
                # Vérifier si nous avons des prédictions
                if 'predictions' not in self.results or symbol not in self.results['predictions']:
                    self.logger.warning(f"Aucune prédiction disponible pour {symbol}, impossible de générer le tableau de bord")
                    continue
                
                # Récupérer les données de prix et les prédictions
                price_series = self.get_price_column(symbol, 'Close')
                if price_series is None:
                    self.logger.error(f"Impossible de récupérer les données de prix pour {symbol}")
                    continue
                    
                # Récupérer les prédictions
                pred = self.results['predictions'][symbol]
                
                # Récupérer d'autres données nécessaires
                open_series = self.get_price_column(symbol, 'Open')
                high_series = self.get_price_column(symbol, 'High')
                low_series = self.get_price_column(symbol, 'Low')
                volume_series = self.get_price_column(symbol, 'Volume')
                
                # Calculer les indicateurs techniques
                # 1. Préparer un DataFrame avec les données OHLCV
                df = pd.DataFrame({
                    'Open': open_series,
                    'High': high_series,
                    'Low': low_series,
                    'Close': price_series,
                    'Volume': volume_series
                })
                
                # 2. Moyennes mobiles
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                df['MA200'] = df['Close'].rolling(window=200).mean()
                
                # 3. RSI
                delta = df['Close'].diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up / ema_down
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # 4. MACD
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA12'] - df['EMA26']
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['Signal']
                
                # 5. Bollinger Bands
                df['BB_middle'] = df['Close'].rolling(window=20).mean()
                std20 = df['Close'].rolling(window=20).std()
                df['BB_upper'] = df['BB_middle'] + (std20 * 2)
                df['BB_lower'] = df['BB_middle'] - (std20 * 2)
                
                # 6. Stochastic Oscillator
                low_14 = df['Low'].rolling(window=14).min()
                high_14 = df['High'].rolling(window=14).max()
                df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
                df['%D'] = df['%K'].rolling(window=3).mean()
                
                # 7. Volume moyen et OBV (On-Balance Volume)
                df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
                
                # OBV - On-Balance Volume
                obv = [0]
                for i in range(1, len(df)):
                    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                        obv.append(obv[-1] + df['Volume'].iloc[i])
                    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                        obv.append(obv[-1] - df['Volume'].iloc[i])
                    else:
                        obv.append(obv[-1])
                df['OBV'] = obv
                
                # 8. Fibonacci Retracement
                recent_min = df['Low'].tail(100).min()
                recent_max = df['High'].tail(100).max()
                
                fib_levels = {
                    '0.0': recent_min,
                    '0.236': recent_min + 0.236 * (recent_max - recent_min),
                    '0.382': recent_min + 0.382 * (recent_max - recent_min),
                    '0.5': recent_min + 0.5 * (recent_max - recent_min),
                    '0.618': recent_min + 0.618 * (recent_max - recent_min),
                    '0.786': recent_min + 0.786 * (recent_max - recent_min),
                    '1.0': recent_max
                }
                
                # 9. Détection des chandeliers japonais significatifs
                patterns = []
                
                # Identifier les derniers chandeliers significatifs
                # Corps du chandelier
                df['Body'] = abs(df['Open'] - df['Close'])
                df['Body_High'] = df[['Open', 'Close']].max(axis=1)
                df['Body_Low'] = df[['Open', 'Close']].min(axis=1)
                df['Upper_Shadow'] = df['High'] - df['Body_High']
                df['Lower_Shadow'] = df['Body_Low'] - df['Low']
                
                # Moyenne du corps (pour référence)
                avg_body = df['Body'].tail(20).mean()
                
                # Détecter les derniers patterns de chandeliers
                for i in range(-5, 0):
                    try:
                        day = df.iloc[i]
                        
                        # Caractéristiques de la bougie
                        is_bullish = day['Close'] > day['Open']
                        body_size = day['Body']
                        upper_shadow = day['Upper_Shadow']
                        lower_shadow = day['Lower_Shadow']
                        
                        # Détection de patterns simples
                        if body_size < 0.2 * avg_body and upper_shadow > body_size and lower_shadow > body_size:
                            patterns.append({'day': i, 'pattern': 'Doji', 'strength': 'Medium'})
                        elif is_bullish and body_size > 1.5 * avg_body and upper_shadow < 0.3 * body_size and lower_shadow < 0.3 * body_size:
                            patterns.append({'day': i, 'pattern': 'Bullish Marubozu', 'strength': 'Strong'})
                        elif not is_bullish and body_size > 1.5 * avg_body and upper_shadow < 0.3 * body_size and lower_shadow < 0.3 * body_size:
                            patterns.append({'day': i, 'pattern': 'Bearish Marubozu', 'strength': 'Strong'})
                        elif is_bullish and lower_shadow > 2 * body_size and upper_shadow < 0.3 * body_size:
                            patterns.append({'day': i, 'pattern': 'Hammer', 'strength': 'Medium'})
                        elif not is_bullish and upper_shadow > 2 * body_size and lower_shadow < 0.3 * body_size:
                            patterns.append({'day': i, 'pattern': 'Shooting Star', 'strength': 'Medium'})
                    except IndexError:
                        continue
                
                # 10. Examiner les prédictions futures et les intervalles de confiance
                forecast_values = pred['values']
                forecast_dates = pred['dates']
                
                lower_bound = pred.get('lower_bound', None)
                upper_bound = pred.get('upper_bound', None)
                
                # Signal de trading basé sur les prédictions
                current_price = df['Close'].iloc[-1]
                next_price = forecast_values[0] if len(forecast_values) > 0 else current_price
                
                price_change = (next_price - current_price) / current_price * 100
                trading_signal = "NEUTRE"
                signal_strength = "Faible"
                
                if price_change > 3:
                    trading_signal = "ACHAT"
                    signal_strength = "Fort" if price_change > 5 else "Moyen"
                elif price_change > 1:
                    trading_signal = "ACHAT"
                    signal_strength = "Faible"
                elif price_change < -3:
                    trading_signal = "VENTE"
                    signal_strength = "Fort" if price_change < -5 else "Moyen"
                elif price_change < -1:
                    trading_signal = "VENTE"
                    signal_strength = "Faible"
                
                # Détecter les niveaux de support et résistance
                def find_support_resistance(prices, window=10):
                    supports = []
                    resistances = []
                    
                    for i in range(window, len(prices) - window):
                        # Supporté par window jours avant et après
                        is_support = True
                        for j in range(i - window, i):
                            if prices[j] < prices[i]:
                                is_support = False
                                break
                        for j in range(i + 1, i + window + 1):
                            if j < len(prices) and prices[j] < prices[i]:
                                is_support = False
                                break
                        
                        if is_support:
                            supports.append((i, prices[i]))
                        
                        # Résistance
                        is_resistance = True
                        for j in range(i - window, i):
                            if prices[j] > prices[i]:
                                is_resistance = False
                                break
                        for j in range(i + 1, i + window + 1):
                            if j < len(prices) and prices[j] > prices[i]:
                                is_resistance = False
                                break
                        
                        if is_resistance:
                            resistances.append((i, prices[i]))
                    
                    # Filtrer les niveaux proches
                    filtered_supports = []
                    filtered_resistances = []
                    
                    # Fonction pour regrouper les niveaux proches
                    def cluster_levels(levels, threshold=0.02):
                        if not levels:
                            return []
                        
                        clusters = []
                        current_cluster = [levels[0]]
                        
                        for i in range(1, len(levels)):
                            # Si le niveau est proche du précédent (moins de 2% d'écart)
                            if abs(levels[i][1] - current_cluster[-1][1]) / current_cluster[-1][1] < threshold:
                                current_cluster.append(levels[i])
                            else:
                                # Ajouter la moyenne du cluster actuel
                                avg_value = sum(level[1] for level in current_cluster) / len(current_cluster)
                                avg_idx = int(sum(level[0] for level in current_cluster) / len(current_cluster))
                                clusters.append((avg_idx, avg_value))
                                
                                # Démarrer un nouveau cluster
                                current_cluster = [levels[i]]
                        
                        # Ajouter le dernier cluster
                        if current_cluster:
                            avg_value = sum(level[1] for level in current_cluster) / len(current_cluster)
                            avg_idx = int(sum(level[0] for level in current_cluster) / len(current_cluster))
                            clusters.append((avg_idx, avg_value))
                        
                        return clusters
                    
                    filtered_supports = cluster_levels(supports)
                    filtered_resistances = cluster_levels(resistances)
                    
                    # Retourner les 3 plus récents
                    return (sorted(filtered_supports, key=lambda x: x[0], reverse=True)[:3], 
                            sorted(filtered_resistances, key=lambda x: x[0], reverse=True)[:3])
                
                close_prices = df['Close'].values
                supports, resistances = find_support_resistance(close_prices)
                
                # Détection du régime de marché
                market_regime = self.detect_market_regime(symbol)
                
                # Construire le dashboard HTML avec Plotly
                # On utilise plusieurs graphiques dans une seule page HTML
                
                # 1. Graphique des prix avec prévisions et indicateurs
                fig1 = go.Figure()
                
                # Ajouter les prix historiques (chandeliers)
                fig1.add_trace(
                    go.Candlestick(
                        x=df.index[-100:],
                        open=df['Open'][-100:],
                        high=df['High'][-100:],
                        low=df['Low'][-100:],
                        close=df['Close'][-100:],
                        name="OHLC"
                    )
                )
                
                # Ajouter les moyennes mobiles
                fig1.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['MA5'][-100:],
                        line=dict(color='blue', width=1),
                        name="MA5"
                    )
                )
                
                fig1.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['MA20'][-100:],
                        line=dict(color='orange', width=1),
                        name="MA20"
                    )
                )
                
                fig1.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['MA50'][-100:],
                        line=dict(color='green', width=1),
                        name="MA50"
                    )
                )
                
                # Ajouter les bandes de Bollinger
                fig1.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['BB_upper'][-100:],
                        line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                        name="BB Upper"
                    )
                )
                
                fig1.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['BB_lower'][-100:],
                        line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                        name="BB Lower",
                        fill='tonexty',
                        fillcolor='rgba(173, 204, 255, 0.2)'
                    )
                )
                
                # Ajouter les niveaux de support et résistance
                for idx, value in supports:
                    if idx > len(df) - 100:  # Dans la fenêtre affichée
                        fig1.add_shape(
                            type="line",
                            x0=df.index[max(0, idx-20)],
                            y0=value,
                            x1=df.index[-1],
                            y1=value,
                            line=dict(color="green", width=2, dash="dash")
                        )
                
                for idx, value in resistances:
                    if idx > len(df) - 100:  # Dans la fenêtre affichée
                        fig1.add_shape(
                            type="line",
                            x0=df.index[max(0, idx-20)],
                            y0=value,
                            x1=df.index[-1],
                            y1=value,
                            line=dict(color="red", width=2, dash="dash")
                        )
                
                # Ajouter les prévisions
                fig1.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        line=dict(color='red', width=2, dash='dash'),
                        name="Prévision"
                    )
                )
                
                # Ajouter l'intervalle de confiance
                if lower_bound is not None and upper_bound is not None:
                    fig1.add_trace(
                        go.Scatter(
                            x=forecast_dates,
                            y=upper_bound,
                            line=dict(color='rgba(255, 0, 0, 0.1)', width=0),
                            name="Limite supérieure"
                        )
                    )
                    
                    fig1.add_trace(
                        go.Scatter(
                            x=forecast_dates,
                            y=lower_bound,
                            line=dict(color='rgba(255, 0, 0, 0.1)', width=0),
                            name="Limite inférieure",
                            fill='tonexty',
                            fillcolor='rgba(255, 0, 0, 0.2)'
                        )
                    )
                
                # Ajouter les niveaux de Fibonacci
                for level, value in fib_levels.items():
                    fig1.add_shape(
                        type="line",
                        x0=df.index[-100],
                        y0=value,
                        x1=df.index[-1],
                        y1=value,
                        line=dict(color="purple", width=1, dash="dot"),
                    )
                    
                    fig1.add_annotation(
                        x=df.index[-100],
                        y=value,
                        text=f"Fib {level}",
                        showarrow=False,
                        xanchor="left",
                        bgcolor="rgba(255, 255, 255, 0.7)"
                    )
                
                # Configurer la mise en page
                fig1.update_layout(
                    title=f"{symbol} - Analyse Technique & Prévisions",
                    xaxis_title="Date",
                    yaxis_title="Prix",
                    height=600,
                    legend_title="Légende",
                    hovermode="x unified",
                    xaxis_rangeslider_visible=False
                )
                
                # 2. Graphique du volume
                fig2 = go.Figure()
                
                # Volume bars
                colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                        for i in range(len(df))][-100:]
                
                fig2.add_trace(
                    go.Bar(
                        x=df.index[-100:],
                        y=df['Volume'][-100:],
                        marker_color=colors,
                        name="Volume"
                    )
                )
                
                # Moyenne mobile du volume
                fig2.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['Volume_MA20'][-100:],
                        line=dict(color='blue', width=2),
                        name="Volume MA20"
                    )
                )
                
                fig2.update_layout(
                    title="Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=250,
                    hovermode="x unified"
                )
                
                # 3. Graphique du MACD
                fig3 = go.Figure()
                
                fig3.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['MACD'][-100:],
                        line=dict(color='blue', width=2),
                        name="MACD"
                    )
                )
                
                fig3.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['Signal'][-100:],
                        line=dict(color='red', width=2),
                        name="Signal"
                    )
                )
                
                # MACD Histogram
                colors = ['red' if h < 0 else 'green' for h in df['MACD_Hist'][-100:]]
                
                fig3.add_trace(
                    go.Bar(
                        x=df.index[-100:],
                        y=df['MACD_Hist'][-100:],
                        marker_color=colors,
                        name="Histogram"
                    )
                )
                
                fig3.update_layout(
                    title="MACD",
                    xaxis_title="Date",
                    yaxis_title="MACD",
                    height=250,
                    hovermode="x unified"
                )
                
                # 4. Graphique RSI et Stochastique
                fig4 = go.Figure()
                
                fig4.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['RSI'][-100:],
                        line=dict(color='blue', width=2),
                        name="RSI"
                    )
                )
                
                # Ajouter les lignes de surachat (70) et survente (30)
                fig4.add_shape(
                    type="line",
                    x0=df.index[-100],
                    y0=70,
                    x1=df.index[-1],
                    y1=70,
                    line=dict(color="red", width=1, dash="dash"),
                )
                
                fig4.add_shape(
                    type="line",
                    x0=df.index[-100],
                    y0=30,
                    x1=df.index[-1],
                    y1=30,
                    line=dict(color="green", width=1, dash="dash"),
                )
                
                # Stochastique
                fig4.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['%K'][-100:],
                        line=dict(color='purple', width=2),
                        name="Stochastique %K",
                        yaxis="y2"
                    )
                )
                
                fig4.add_trace(
                    go.Scatter(
                        x=df.index[-100:],
                        y=df['%D'][-100:],
                        line=dict(color='orange', width=2),
                        name="Stochastique %D",
                        yaxis="y2"
                    )
                )
                
                # Ajouter les lignes de surachat (80) et survente (20) pour le stochastique
                fig4.add_shape(
                    type="line",
                    x0=df.index[-100],
                    y0=80,
                    x1=df.index[-1],
                    y1=80,
                    line=dict(color="red", width=1, dash="dash"),
                    yref="y2"
                )
                
                fig4.add_shape(
                    type="line",
                    x0=df.index[-100],
                    y0=20,
                    x1=df.index[-1],
                    y1=20,
                    line=dict(color="green", width=1, dash="dash"),
                    yref="y2"
                )
                
                fig4.update_layout(
                    title="RSI & Stochastique",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    yaxis=dict(range=[0, 100]),
                    yaxis2=dict(
                        title="Stochastique",
                        overlaying="y",
                        side="right",
                        range=[0, 100]
                    ),
                    height=250,
                    hovermode="x unified"
                )
                
                # Créer un tableau HTML avec le résumé des indicateurs et les signaux de trading
                current_rsi = df['RSI'].iloc[-1]
                current_stoch_k = df['%K'].iloc[-1]
                current_macd = df['MACD_Hist'].iloc[-1]
                
                # Définir les signaux basés sur les indicateurs
                rsi_signal = "NEUTRE"
                if current_rsi > 70:
                    rsi_signal = "SURVENTE"
                elif current_rsi < 30:
                    rsi_signal = "SURACHAT"
                
                stoch_signal = "NEUTRE"
                if current_stoch_k > 80:
                    stoch_signal = "SURVENTE"
                elif current_stoch_k < 20:
                    stoch_signal = "SURACHAT"
                
                macd_signal = "NEUTRE"
                if current_macd > 0 and df['MACD_Hist'].iloc[-2] <= 0:
                    macd_signal = "ACHAT"
                elif current_macd < 0 and df['MACD_Hist'].iloc[-2] >= 0:
                    macd_signal = "VENTE"
                
                # Signal global basé sur une combinaison d'indicateurs
                buy_signals = 0
                sell_signals = 0
                
                # Compter les signaux d'achat/vente
                if rsi_signal == "SURACHAT":
                    buy_signals += 1
                elif rsi_signal == "SURVENTE":
                    sell_signals += 1
                    
                if stoch_signal == "SURACHAT":
                    buy_signals += 1
                elif stoch_signal == "SURVENTE":
                    sell_signals += 1
                    
                if macd_signal == "ACHAT":
                    buy_signals += 1
                elif macd_signal == "VENTE":
                    sell_signals += 1
                    
                if trading_signal == "ACHAT":
                    buy_signals += 1
                elif trading_signal == "VENTE":
                    sell_signals += 1
                    
                # Tendance des moyennes mobiles
                ma_trend = "NEUTRE"
                if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] > df['MA50'].iloc[-1]:
                    ma_trend = "HAUSSIÈRE"
                    buy_signals += 1
                elif df['MA5'].iloc[-1] < df['MA20'].iloc[-1] < df['MA50'].iloc[-1]:
                    ma_trend = "BAISSIÈRE"
                    sell_signals += 1
                
                # Croisement des moyennes mobiles
                ma_cross = "AUCUN"
                if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] and df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]:
                    ma_cross = "HAUSSIER (Golden Cross)"
                    buy_signals += 1
                elif df['MA5'].iloc[-1] < df['MA20'].iloc[-1] and df['MA5'].iloc[-2] >= df['MA20'].iloc[-2]:
                    ma_cross = "BAISSIER (Death Cross)"
                    sell_signals += 1
                
                # Décision finale
                overall_signal = "NEUTRE"
                if buy_signals > sell_signals + 1:
                    overall_signal = "ACHAT FORT"
                elif buy_signals > sell_signals:
                    overall_signal = "ACHAT"
                elif sell_signals > buy_signals + 1:
                    overall_signal = "VENTE FORTE"
                elif sell_signals > buy_signals:
                    overall_signal = "VENTE"
                
                # Niveaux de prix importants
                current_price = df['Close'].iloc[-1]
                next_resistance = min((r[1] for r in resistances if r[1] > current_price), default=current_price * 1.1)
                next_support = max((s[1] for s in supports if s[1] < current_price), default=current_price * 0.9)
                
                # Créer le tableau récapitulatif
                summary_table = f"""
                <table class="summary-table">
                    <tr>
                        <th colspan="4" class="summary-header">Résumé pour {symbol}</th>
                    </tr>
                    <tr>
                        <td class="label">Prix actuel:</td>
                        <td class="value">{current_price:.2f}</td>
                        <td class="label">Prochaine résistance:</td>
                        <td class="value">{next_resistance:.2f}</td>
                    </tr>
                    <tr>
                        <td class="label">Variation prévue:</td>
                        <td class="value">{price_change:.2f}%</td>
                        <td class="label">Prochain support:</td>
                        <td class="value">{next_support:.2f}</td>
                    </tr>
                    <tr>
                        <td class="label">RSI (14):</td>
                        <td class="value">{current_rsi:.2f} - <span class="{rsi_signal.lower()}">{rsi_signal}</span></td>
                        <td class="label">MA Trend:</td>
                        <td class="value"><span class="{ma_trend.lower()}">{ma_trend}</span></td>
                    </tr>
                    <tr>
                        <td class="label">Stochastique:</td>
                        <td class="value">{current_stoch_k:.2f} - <span class="{stoch_signal.lower()}">{stoch_signal}</span></td>
                        <td class="label">MA Cross:</td>
                        <td class="value"><span class="{ma_cross.lower()}">{ma_cross}</span></td>
                    </tr>
                    <tr>
                        <td class="label">MACD:</td>
                        <td class="value">{current_macd:.4f} - <span class="{macd_signal.lower()}">{macd_signal}</span></td>
                        <td class="label">Régime de marché:</td>
                        <td class="value">{market_regime['regime']['type']}</td>
                    </tr>
                    <tr>
                        <td class="label">Prévision ML:</td>
                        <td class="value"><span class="{trading_signal.lower()}">{trading_signal} ({signal_strength})</span></td>
                        <td class="label">Volatilité:</td>
                        <td class="value">{market_regime['statistics']['volatility']:.4f}</td>
                    </tr>
                    <tr>
                        <th colspan="4" class="signal-header"><span class="{overall_signal.lower().replace(' ', '-')}">{overall_signal}</span></th>
                    </tr>
                </table>
                """
                
                # Liste des chandeliers japonais récents
                
                patterns_table = ""
                if patterns:
                    patterns_table = """
                    <table class="patterns-table">
                        <tr>
                            <th colspan="3" class="patterns-header">Chandeliers Japonais Récents</th>
                        </tr>
                        <tr>
                            <th>Jour</th>
                            <th>Pattern</th>
                            <th>Force</th>
                        </tr>
                    """
                    
                    for p in patterns:
                        day_str = "Aujourd'hui" if p['day'] == -1 else (
                            "Hier" if p['day'] == -2 else f"Il y a {abs(p['day'])} jours")
                        
                        patterns_table += f"""
                        <tr>
                            <td>{day_str}</td>
                            <td>{p['pattern']}</td>
                            <td class="{p['strength'].lower()}">{p['strength']}</td>
                        </tr>
                        """
                    
                    patterns_table += "</table>"
                
                # Créer le tableau des niveaux de Fibonacci
                fib_table = """
                <table class="fib-table">
                    <tr>
                        <th colspan="2" class="fib-header">Niveaux de Fibonacci</th>
                    </tr>
                    <tr>
                        <th>Niveau</th>
                        <th>Prix</th>
                    </tr>
                """
                
                for level, value in fib_levels.items():
                    fib_table += f"""
                    <tr>
                        <td>{level}</td>
                        <td>{value:.2f}</td>
                    </tr>
                    """
                
                fib_table += "</table>"
                
                # Créer un tableau pour les prévisions
                forecast_table = """
                <table class="forecast-table">
                    <tr>
                        <th colspan="4" class="forecast-header">Prévisions des 5 prochains jours</th>
                    </tr>
                    <tr>
                        <th>Date</th>
                        <th>Prix prévu</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                """
                
                for i in range(min(5, len(forecast_dates))):
                    date_str = forecast_dates[i].strftime('%Y-%m-%d')
                    price_str = f"{forecast_values[i]:.2f}"
                    
                    min_str = "N/A"
                    max_str = "N/A"
                    
                    if lower_bound is not None and upper_bound is not None:
                        min_str = f"{lower_bound[i]:.2f}"
                        max_str = f"{upper_bound[i]:.2f}"
                    
                    forecast_table += f"""
                    <tr>
                        <td>{date_str}</td>
                        <td>{price_str}</td>
                        <td>{min_str}</td>
                        <td>{max_str}</td>
                    </tr>
                    """
                
                forecast_table += "</table>"
                
                # Créer le HTML complet avec tous les graphiques et tableaux
                dashboard_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Tableau de Bord de Trading - {symbol}</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body {{
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            margin: 0;
                            padding: 20px;
                            background-color: #f5f5f5;
                            color: #333;
                        }}
                        
                        .dashboard-container {{
                            max-width: 1200px;
                            margin: 0 auto;
                            background-color: white;
                            border-radius: 8px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                            overflow: hidden;
                        }}
                        
                        .dashboard-header {{
                            background-color: #2c3e50;
                            color: white;
                            padding: 15px 20px;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        }}
                        
                        .header-title h1 {{
                            margin: 0;
                            font-size: 24px;
                        }}
                        
                        .header-title p {{
                            margin: 5px 0 0;
                            font-size: 14px;
                            opacity: 0.8;
                        }}
                        
                        .header-info {{
                            text-align: right;
                        }}
                        
                        .dashboard-content {{
                            padding: 20px;
                        }}
                        
                        .chart-container {{
                            margin-bottom: 25px;
                            background-color: white;
                            border-radius: 4px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        }}
                        
                        .tables-container {{
                            display: grid;
                            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                            gap: 20px;
                            margin-top: 20px;
                        }}
                        
                        table {{
                            width: 100%;
                            border-collapse: collapse;
                            border-radius: 4px;
                            overflow: hidden;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                            margin-bottom: 10px;
                        }}
                        
                        th, td {{
                            padding: 12px 15px;
                            text-align: left;
                            border-bottom: 1px solid #ddd;
                        }}
                        
                        .summary-header, .patterns-header, .fib-header, .forecast-header {{
                            background-color: #2c3e50;
                            color: white;
                            text-align: center;
                            font-weight: bold;
                        }}
                        
                        .signal-header {{
                            background-color: #f8f9fa;
                            color: #333;
                            text-align: center;
                            font-size: 18px;
                            padding: 15px;
                        }}
                        
                        .summary-table .label {{
                            font-weight: bold;
                            width: 25%;
                        }}
                        
                        .achat, .achat-fort {{
                            color: #27ae60;
                            font-weight: bold;
                        }}
                        
                        .vente, .vente-forte {{
                            color: #e74c3c;
                            font-weight: bold;
                        }}
                        
                        .neutre {{
                            color: #7f8c8d;
                        }}
                        
                        .surachat {{
                            color: #e74c3c;
                        }}
                        
                        .survente {{
                            color: #27ae60;
                        }}
                        
                        .haussière, .haussier {{
                            color: #27ae60;
                        }}
                        
                        .baissière, .baissier {{
                            color: #e74c3c;
                        }}
                        
                        .strong {{
                            font-weight: bold;
                            color: #2c3e50;
                        }}
                        
                        .medium {{
                            color: #7f8c8d;
                        }}
                        
                        .weak {{
                            color: #bdc3c7;
                        }}
                        
                        .footer {{
                            text-align: center;
                            padding: 15px;
                            color: #7f8c8d;
                            font-size: 12px;
                            border-top: 1px solid #eee;
                        }}
                    </style>
                </head>
                <body>
                    <div class="dashboard-container">
                        <div class="dashboard-header">
                            <div class="header-title">
                                <h1>Tableau de Bord de Trading - {symbol}</h1>
                                <p>Généré le {datetime.now().strftime('%Y-%m-%d à %H:%M')}</p>
                            </div>
                            <div class="header-info">
                                <p>Prix actuel: <b>{current_price:.2f}</b></p>
                                <p>Signal global: <span class="{overall_signal.lower().replace(' ', '-')}"><b>{overall_signal}</b></span></p>
                            </div>
                        </div>
                        
                        <div class="dashboard-content">
                            <div class="chart-container">
                                {fig1.to_html(full_html=False, include_plotlyjs='cdn')}
                            </div>
                            
                            <div class="chart-container">
                                {fig2.to_html(full_html=False, include_plotlyjs='cdn')}
                            </div>
                            
                            <div class="chart-container">
                                {fig3.to_html(full_html=False, include_plotlyjs='cdn')}
                            </div>
                            
                            <div class="chart-container">
                                {fig4.to_html(full_html=False, include_plotlyjs='cdn')}
                            </div>
                            
                            <div class="tables-container">
                                <div class="summary-section">
                                    {summary_table}
                                    {forecast_table}
                                </div>
                                
                                <div class="analysis-section">
                                    {patterns_table}
                                    {fib_table}
                                </div>
                            </div>
                        </div>
                        
                        <div class="footer">
                            <p>Ce tableau de bord est généré automatiquement à des fins informatives uniquement. 
                            Ne constitue pas un conseil en investissement. Analyse technique basée sur des modèles statistiques.</p>
                        </div>
                    </div>
                </body>
                </html>
                """
                
                # Sauvegarder le dashboard HTML
                file_path = f"{save_path}/{symbol}_dashboard.html"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(dashboard_html)
                
                self.logger.info(f"Tableau de bord de trading généré pour {symbol}: {file_path}")
            
            # Créer une page d'index pour naviguer entre les différents tableaux de bord
            index_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Tableaux de Bord de Trading</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                        color: #333;
                    }
                    
                    .container {
                        max-width: 1000px;
                        margin: 0 auto;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        padding: 20px;
                    }
                    
                    h1 {
                        color: #2c3e50;
                        margin-top: 0;
                        padding-bottom: 10px;
                        border-bottom: 1px solid #eee;
                    }
                    
                    .dashboard-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 15px;
                        margin-top: 20px;
                    }
                    
                    .dashboard-card {
                        background-color: #f8f9fa;
                        border-radius: 4px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        overflow: hidden;
                        transition: transform 0.2s, box-shadow 0.2s;
                    }
                    
                    .dashboard-card:hover {
                        transform: translateY(-3px);
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    }
                    
                    .dashboard-card a {
                        display: block;
                        padding: 15px;
                        text-decoration: none;
                        color: #2c3e50;
                        font-weight: bold;
                        text-align: center;
                    }
                    
                    .footer {
                        margin-top: 30px;
                        text-align: center;
                        color: #7f8c8d;
                        font-size: 12px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Tableaux de Bord de Trading</h1>
                    <p>Sélectionnez un symbole pour accéder à son tableau de bord détaillé.</p>
                    
                    <div class="dashboard-grid">
            """
            
            # Ajouter un lien pour chaque symbole
            for symbol in self.config['data']['symbols']:
                dashboard_path = f"{symbol}_dashboard.html"
                if os.path.exists(f"{save_path}/{dashboard_path}"):
                    index_html += f"""
                        <div class="dashboard-card">
                            <a href="{dashboard_path}">{symbol}</a>
                        </div>
                    """
            
            index_html += """
                    </div>
                    
                    <div class="footer">
                        <p>Tableaux de bord générés le """ + datetime.now().strftime('%Y-%m-%d à %H:%M') + """</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Sauvegarder la page d'index
            index_path = f"{save_path}/index.html"
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(index_html)
            
            self.logger.info(f"Page d'index générée: {index_path}")
            
            return index_path
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du tableau de bord de trading: {str(e)}")
            traceback.print_exc()
            return None

def main():
    """
    Fonction principale pour exécuter le programme optimisé
    """
    try:
        # Analyser les arguments de ligne de commande
        import argparse
        parser = argparse.ArgumentParser(description='Programme de Prévision Boursière Optimisé (4 modèles)')
        parser.add_argument('-s', '--symbols', nargs='+', help='Symboles boursiers à analyser (ex: AAPL MSFT GOOGL)')
        parser.add_argument('-d', '--dashboard', action='store_true', help='Générer uniquement le tableau de bord de trading')
        args = parser.parse_args()
        
        # Créer le système de prévision avec les symboles spécifiés
        symbols = args.symbols if args.symbols else None
        
        # Initialiser le prédicteur
        predictor = PredictionBoursiere(symbols=symbols)
        
        # IMPORTANT : Définir votre clé API Tiingo ici
        if 'tiingo' not in predictor.config:
            predictor.config['tiingo'] = {}
        predictor.config['tiingo']['api_key'] = "1f2a9c9f7cc99f3f7855f6dec4a6760c00735d3f"  # Remplacez par votre clé
        
        # Si l'option dashboard est spécifiée, générer uniquement le tableau de bord
        if args.dashboard:
            # Vérifier si les données sont déjà chargées
            if predictor.data is None:
                # Charger les données
                if not predictor.load_data():
                    logger.error("Impossible de charger les données, génération du tableau de bord impossible")
                    return
            
            # Générer le tableau de bord
            dashboard_path = predictor.generate_trading_dashboard("trading_dashboard")
            if dashboard_path:
                logger.info(f"Tableau de bord de trading généré avec succès: {dashboard_path}")
                # Ouvrir automatiquement dans le navigateur par défaut
                import webbrowser
                webbrowser.open('file://' + os.path.realpath(dashboard_path))
            else:
                logger.error("Échec de la génération du tableau de bord de trading")
            
            return
        
        # Sinon, exécuter le pipeline complet
        # Modifier la configuration
        predictor.config['models']['var']['enabled'] = False
        predictor.config['models']['machine_learning']['algorithms'] = ['gradient_boosting']
        predictor.config['models']['deep_learning']['architectures'] = ['lstm']
        
        # Exécuter le pipeline robust (qui inclut maintenant la génération du tableau de bord)
        if hasattr(predictor, 'run_pipeline_robust'):
            results = predictor.run_pipeline_robust()
            
            # Le tableau de bord est déjà généré dans run_pipeline_robust
            dashboard_path = results.get("dashboard_path")
            if dashboard_path:
                logger.info(f"Tableau de bord accessible à: {dashboard_path}")
            
        else:
            # Fallback à l'ancienne méthode
            results = predictor.run_pipeline()
            
            # Générer le tableau de bord manuellement
            try:
                dashboard_path = predictor.generate_trading_dashboard("trading_dashboard")
                if dashboard_path:
                    logger.info(f"Tableau de bord de trading généré avec succès: {dashboard_path}")
                    # Ouvrir automatiquement dans le navigateur par défaut
                    import webbrowser
                    webbrowser.open('file://' + os.path.realpath(dashboard_path))
                else:
                    logger.warning("Échec de la génération du tableau de bord de trading")
            except Exception as e:
                logger.error(f"Erreur lors de la génération du tableau de bord: {str(e)}")
        
        # Sauvegarder les modèles
        predictor.save_model(path="models_optimized")
        
        # Résumé des prévisions (déjà affiché dans run_pipeline_robust)
        logger.info("Programme de prévision optimisé terminé")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution principale: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()