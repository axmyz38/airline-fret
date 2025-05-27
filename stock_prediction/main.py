import yaml
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    """Point d'entrée principal du système optimisé"""
    
    # Arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Système de Prévision Boursière Optimisé')
    parser.add_argument('-c', '--config', type=str, default='config/config.yaml',
                       help='Chemin vers le fichier de configuration')
    parser.add_argument('-s', '--symbols', nargs='+', 
                       help='Symboles à analyser')
    parser.add_argument('-m', '--mode', choices=['train', 'predict', 'backtest', 'dashboard'],
                       default='train', help='Mode d\'exécution')
    parser.add_argument('--parallel', action='store_true',
                       help='Activer la parallélisation')
    args = parser.parse_args()
    
    # Charger la configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Mettre à jour avec les arguments
    if args.symbols:
        config['data']['symbols'] = args.symbols
    
    # Configurer le logger
    logger = setup_logger(config)
    logger.info("Démarrage du système de prévision boursière optimisé")
    
    try:
        # Initialiser les composants
        from src.data.data_loader import DataLoader
        from src.features.feature_engineering import FeatureEngineer
        from src.models.model_factory import ModelFactory
        from src.backtest.backtester import WalkForwardBacktester
        from src.visualization.dashboard import TradingDashboard
        
        # Charger les données
        data_loader = DataLoader(config)
        data = data_loader.load_data()
        
        # Feature engineering
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.create_features(data)
        
        if args.mode == 'train':
            # Entraîner les modèles
            model_factory = ModelFactory(config)
            models = model_factory.create_models()
            
            # Entraînement parallèle si demandé
            if args.parallel:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor() as executor:
                    futures = []
                    for symbol in config['data']['symbols']:
                        future = executor.submit(train_symbol_models, 
                                               symbol, models, features, config)
                        futures.append(future)
                    
                    for future in futures:
                        future.result()
            else:
                for symbol in config['data']['symbols']:
                    train_symbol_models(symbol, models, features, config)
            
            logger.info("Entraînement terminé")
            
        elif args.mode == 'predict':
            # Générer des prévisions
            model_factory = ModelFactory(config)
            models = model_factory.load_models()
            
            predictions = {}
            for symbol in config['data']['symbols']:
                ensemble = EnsembleModel(models[symbol], config)
                pred = ensemble.predict(features[symbol], config['prediction']['horizon'])
                predictions[symbol] = pred
            
            # Sauvegarder les prévisions
            save_predictions(predictions, config)
            
        elif args.mode == 'backtest':
            # Backtesting
            model_factory = ModelFactory(config)
            models = model_factory.load_models()
            
            backtester = WalkForwardBacktester(models, config)
            results = backtester.run_backtest(features)
            
            # Visualiser et sauvegarder les résultats
            backtester.plot_results(results, 'backtest_results.png')
            
        elif args.mode == 'dashboard':
            # Dashboard interactif
            dashboard = TradingDashboard(config)
            dashboard.run()
            
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {str(e)}", exc_info=True)
        raise

def train_symbol_models(symbol, models, features, config):
    """Entraîne tous les modèles pour un symbole"""
    logger = logging.getLogger(__name__)
    logger.info(f"Entraînement des modèles pour {symbol}")
    
    X = features[symbol]
    y = X['target']
    X = X.drop('target', axis=1)
    
    trained_models = {}
    for model_name, model in models.items():
        try:
            logger.info(f"Entraînement de {model_name} pour {symbol}")
            model.fit(X, y)
            trained_models[model_name] = model
            
            # Sauvegarder le modèle
            model.save(f"models/{symbol}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement de {model_name}: {e}")
    
    return trained_models

def save_predictions(predictions, config):
    """Sauvegarde les prévisions"""
    import json
    from datetime import datetime
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'predictions': {}
    }
    
    for symbol, pred in predictions.items():
        output['predictions'][symbol] = {
            'values': pred.tolist(),
            'horizon': len(pred)
        }
    
    with open('predictions.json', 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()

# ========== config/config.yaml ==========
"""
data:
  symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
  start_date: "2020-01-01"
  end_date: "2024-01-01"
  api_key: "your_tiingo_api_key"
  cache_dir: "data/cache"

features:
  technical_indicators:
    enabled: true
    indicators: ["MA", "EMA", "RSI", "MACD", "BB", "STOCH", "ADX", "ATR", "CCI", "WILLR", "OBV"]
  sentiment:
    enabled: true
    sources: ["twitter", "news", "reddit"]
  calendar:
    enabled: true
    features: ["day_of_week", "month_end", "quarter_end", "holidays"]
  
models:
  arima:
    enabled: true
    auto_arima: true
    seasonal: true
    max_p: 5
    max_q: 5
  
  garch:
    enabled: true
    p: 1
    q: 1
    distribution: "t"
  
  machine_learning:
    enabled: true
    algorithms:
      gradient_boosting:
        n_estimators: 100
        max_depth: 5
        learning_rate: 0.1
      random_forest:
        n_estimators: 200
        max_depth: 10
      xgboost:
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.3
  
  deep_learning:
    enabled: true
    architectures:
      lstm:
        layers: [128, 64, 32]
        dropout: 0.2
        sequence_length: 30
      transformer:
        d_model: 128
        n_heads: 8
        n_layers: 4

optimization:
  method: "optuna"
  n_trials: 100
  timeout: 3600
  parallel: true

backtest:
  initial_train_size: 0.7
  step_size: 20
  recalibrate_every: 60
  transaction_cost: 0.001
  
prediction:
  horizon: 10
  confidence_level: 0.95
  
ensemble:
  method: "stacking"  # "stacking" or "blending"
  meta_model: "ridge"
  
visualization:
  dashboard: true
  port: 8050
  update_interval: 60  # seconds

logging:
  level: "INFO"
  file: "logs/stock_prediction.log"
"""