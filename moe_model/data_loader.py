# moe_model/data_loader.py
import pandas as pd
import numpy as np
import os

def load_and_prepare_data(config):
    """Loads data, handles basic cleaning and target definition."""
    file_path = config["csv_file"]
    target_col = config["target_column"]
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Creating dummy data.")
        # Dummy data creation (as before)
        dates = pd.date_range(start='2019-01-01', periods=1000, freq='B')
        data = np.random.rand(1000, 5) * 100 + 50
        df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=dates)
        df['Adj Close'] = df['Close']; df['Volume'] *= 10000
        data_dir = os.path.dirname(file_path)
        if data_dir and not os.path.exists(data_dir): os.makedirs(data_dir)
        df.to_csv(file_path); print(f"Dummy data saved to {file_path}")

    # Basic Column Checks & Handling Adj Close
    if target_col == 'Adj Close' and 'Adj Close' in df.columns:
        print("Using 'Adj Close' as target.")
        base_price_col = 'Adj Close'
        # Ensure standard OHLC exist, approximate if needed
        if 'Close' not in df.columns: df['Close'] = df['Adj Close']
        if 'Open' not in df.columns: df['Open'] = df['Adj Close'].shift(1)
        if 'High' not in df.columns: df['High'] = df['Adj Close']
        if 'Low' not in df.columns: df['Low'] = df['Adj Close']
    elif 'Close' in df.columns:
        print("Using 'Close' as target."); base_price_col = 'Close'
        if 'Adj Close' in df.columns: df = df.drop(columns=['Adj Close'])
    else: raise ValueError(f"Target column '{target_col}' or 'Close' not found.")

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' missing. Approximating from '{base_price_col}'.")
            if col == 'Open': df[col] = df[base_price_col].shift(1)
            elif col == 'Volume': df[col] = 1e6
            else: df[col] = df[base_price_col]

    # Handle initial NaNs and Infs before feature engineering
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True); df.fillna(method='bfill', inplace=True)
    df['Volume'] = df['Volume'].fillna(0)

    # Define Target column *before* feature engineering that might use it indirectly
    # Ensure target is based on the *original* price column before potential TA modifications
    if base_price_col not in df.columns: # Should not happen based on checks above
         raise ValueError(f"Base price column '{base_price_col}' lost during initial processing.")
    df['Target'] = df[base_price_col].shift(-1)

    print(f"Data loaded. Shape before features: {df.shape}")
    return df, base_price_col