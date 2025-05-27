from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@dataclass
class BacktestResults:
    """Résultats structurés du backtesting"""
    returns: np.ndarray
    predictions: np.ndarray
    actual: np.ndarray
    dates: pd.DatetimeIndex
    metrics: Dict[str, float]
    positions: np.ndarray
    equity_curve: np.ndarray

class WalkForwardBacktester:
    """Backtesting avec walk-forward analysis et recalibrage"""
    
    def __init__(self, models: Dict[str, BaseModel], config: Dict[str, Any]):
        self.models = models
        self.config = config
        self.results = {}
        
    def run_backtest(self, data: pd.DataFrame, 
                     initial_train_size: float = 0.7,
                     step_size: int = 20,
                     recalibrate_every: int = 60) -> Dict[str, BacktestResults]:
        """
        Execute walk-forward backtest avec recalibrage périodique
        """
        results = {}
        
        n_samples = len(data)
        train_size = int(n_samples * initial_train_size)
        
        for model_name, model in self.models.items():
            predictions = []
            actual_values = []
            dates = []
            positions = []
            
            # Walk-forward loop
            for i in range(train_size, n_samples - self.config['horizon'], step_size):
                # Recalibrer le modèle si nécessaire
                if (i - train_size) % recalibrate_every == 0:
                    train_data = data.iloc[:i]
                    X_train = train_data.drop('target', axis=1)
                    y_train = train_data['target']
                    model.fit(X_train, y_train)
                
                # Prédire
                test_idx = min(i + step_size, n_samples - self.config['horizon'])
                X_test = data.iloc[i:test_idx].drop('target', axis=1)
                y_test = data.iloc[i:test_idx]['target']
                
                pred = model.predict(X_test, horizon=len(X_test))
                
                predictions.extend(pred)
                actual_values.extend(y_test.values)
                dates.extend(y_test.index)
                
                # Calculer les positions (long/short/neutre)
                for j in range(len(pred)):
                    if pred[j] > y_test.iloc[j] * 1.01:  # Prévision > 1%
                        positions.append(1)  # Long
                    elif pred[j] < y_test.iloc[j] * 0.99:  # Prévision < -1%
                        positions.append(-1)  # Short
                    else:
                        positions.append(0)  # Neutre
            
            # Calculer les métriques
            predictions = np.array(predictions)
            actual_values = np.array(actual_values)
            positions = np.array(positions)
            
            # Returns
            returns = np.diff(actual_values) / actual_values[:-1]
            strategy_returns = returns * positions[:-1]
            
            # Equity curve
            equity_curve = np.cumprod(1 + strategy_returns)
            
            # Métriques de performance
            metrics = self._calculate_metrics(predictions, actual_values, 
                                            strategy_returns, equity_curve)
            
            results[model_name] = BacktestResults(
                returns=strategy_returns,
                predictions=predictions,
                actual=actual_values,
                dates=pd.DatetimeIndex(dates),
                metrics=metrics,
                positions=positions,
                equity_curve=equity_curve
            )
        
        return results
    
    def _calculate_metrics(self, predictions: np.ndarray, 
                          actual: np.ndarray,
                          returns: np.ndarray,
                          equity_curve: np.ndarray) -> Dict[str, float]:
        """Calcule toutes les métriques de performance"""
        
        # Métriques de prédiction
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        # Direction accuracy
        actual_direction = np.sign(np.diff(actual))
        pred_direction = np.sign(np.diff(predictions))
        direction_accuracy = np.mean(actual_direction == pred_direction)
        
        # Métriques de trading
        total_return = equity_curve[-1] - 1 if len(equity_curve) > 0 else 0
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Sharpe ratio
        sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning_trades = np.sum(returns > 0)
        total_trades = np.sum(returns != 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'num_trades': total_trades
        }
    
    def plot_results(self, results: Dict[str, BacktestResults], save_path: Optional[str] = None):
        """Visualise les résultats du backtesting"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity curves
        ax1 = axes[0, 0]
        for model_name, result in results.items():
            ax1.plot(result.dates, result.equity_curve, label=model_name)
        ax1.set_title('Equity Curves')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Predictions vs Actual
        ax2 = axes[0, 1]
        best_model = max(results.items(), key=lambda x: x[1].metrics['sharpe_ratio'])
        ax2.scatter(best_model[1].actual, best_model[1].predictions, alpha=0.5)
        ax2.plot([min(best_model[1].actual), max(best_model[1].actual)],
                [min(best_model[1].actual), max(best_model[1].actual)], 
                'r--', label='Perfect prediction')
        ax2.set_title(f'Predictions vs Actual - {best_model[0]}')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Returns distribution
        ax3 = axes[1, 0]
        for model_name, result in results.items():
            ax3.hist(result.returns, bins=50, alpha=0.5, label=model_name)
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Returns')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Performance metrics
        ax4 = axes[1, 1]
        metrics_df = pd.DataFrame({name: result.metrics 
                                  for name, result in results.items()}).T
        metrics_df[['sharpe_ratio', 'calmar_ratio', 'direction_accuracy']].plot(
            kind='bar', ax=ax4)
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Value')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
