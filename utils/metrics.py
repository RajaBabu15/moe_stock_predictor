# utils/metrics.py
import numpy as np
import pandas as pd

def calculate_financial_metrics(y_true_unscaled, y_pred_unscaled):
    """Calculates Sharpe, Sortino (basic), Max Drawdown, Calmar."""
    if len(y_true_unscaled) < 2 or len(y_pred_unscaled) < 2 : return {}

    y_true = pd.Series(y_true_unscaled.flatten())
    y_pred = pd.Series(y_pred_unscaled.flatten())

    # Calculate returns based on predicted prices relative to previous *actual* price
    # This simulates a strategy where you trade based on prediction vs yesterday's close
    strategy_returns = y_pred[1:].values / y_true[:-1].values - 1
    strategy_returns = pd.Series(strategy_returns).fillna(0)

    metrics = {}
    epsilon = 1e-9 # Avoid division by zero

    # Sharpe Ratio (annualized)
    if strategy_returns.std() > epsilon:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        metrics['sharpe_ratio'] = sharpe
    else:
        metrics['sharpe_ratio'] = 0.0

    # Sortino Ratio (annualized) - uses downside deviation
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = downside_returns.std()
    if downside_std > epsilon:
        sortino = (strategy_returns.mean() / downside_std) * np.sqrt(252)
        metrics['sortino_ratio'] = sortino
    else:
         # Handle case with no downside returns or zero std dev
        metrics['sortino_ratio'] = np.inf if strategy_returns.mean() > 0 else 0.0


    # Max Drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    metrics['max_drawdown'] = max_drawdown if pd.notna(max_drawdown) else 0.0 # Handle potential NaN

    # Calmar Ratio (annualized return / abs(max drawdown))
    annualized_return = strategy_returns.mean() * 252
    if abs(max_drawdown) > epsilon:
        calmar = annualized_return / abs(max_drawdown)
        metrics['calmar_ratio'] = calmar
    else:
        metrics['calmar_ratio'] = np.inf if annualized_return > 0 else 0.0


    print(f"Financial Metrics: { {k: f'{v:.3f}' for k, v in metrics.items()} }")
    return metrics