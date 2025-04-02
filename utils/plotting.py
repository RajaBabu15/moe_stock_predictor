# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, config):
    """Plots training and validation loss, including auxiliary losses."""
    plt.figure(figsize=(15, 7))
    loss_keys = [k for k in history.history.keys() if 'loss' in k and 'val_' not in k]
    val_loss_keys = [k for k in history.history.keys() if 'loss' in k and 'val_' in k]

    for key in loss_keys:
        label = key.replace('_', ' ').title()
        linestyle = '-'
        scale = 1.0
        # Specific handling for load balancing loss visualization
        if 'load_balancing' in key:
             lb_weight = config.get("load_balancing_loss_weight", 0.01)
             scale = 1.0 / lb_weight if lb_weight > 0 else 1.0
             label = f'LB Loss (x{scale:.0f})' # Modify label
             linestyle = ':'
        plt.plot(np.array(history.history[key]) * scale, label=f'Train {label}', linestyle=linestyle)

    for key in val_loss_keys:
        label = key.replace('val_', '').replace('_', ' ').title()
        linestyle = '-'
        scale = 1.0
        if 'load_balancing' in key:
            lb_weight = config.get("load_balancing_loss_weight", 0.01)
            scale = 1.0 / lb_weight if lb_weight > 0 else 1.0
            label = f'LB Loss (x{scale:.0f})'
            linestyle = ':'
        plt.plot(np.array(history.history[key]) * scale, label=f'Val {label}', linestyle=linestyle)


    plt.title(f'Model Training History')
    plt.xlabel('Epoch'); plt.ylabel('Loss (Scaled where applicable)'); plt.legend(); plt.grid(True); plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

def plot_predictions(y_test_unscaled, y_pred_mean, y_pred_std, config, target_col_name):
    """Plots actual vs predicted prices with optional uncertainty bands."""
    plt.figure(figsize=(15, 7))
    plt.plot(y_test_unscaled, label='Actual Prices', color='blue', alpha=0.8, linewidth=1.5)
    plt.plot(y_pred_mean, label='Predicted Prices (Mean)', color='red', linestyle='-', linewidth=1.5)

    if y_pred_std is not None and config.get("predict_uncertainty", False):
        # Ensure std dev is positive
        y_pred_std = np.maximum(y_pred_std, 1e-9) # Floor std dev to avoid issues
        lower_bound = (y_pred_mean - 1.96 * y_pred_std).flatten()
        upper_bound = (y_pred_mean + 1.96 * y_pred_std).flatten()
        plt.fill_between(range(len(y_pred_mean)), lower_bound, upper_bound,
                         color='orange', alpha=0.3, label='95% Confidence Interval (MC Dropout)')

    plt.title(f'Stock Price Prediction vs Actual')
    plt.xlabel('Time Steps (Test Set)'); plt.ylabel(f'Stock Price ({target_col_name})')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()