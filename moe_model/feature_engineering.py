# moe_model/feature_engineering.py
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.volatility import AverageTrueRange # <<< Import ATR

def add_enhanced_features(df, config):
    """Adds volatility, FFT, range, and ATR features."""
    print("Adding enhanced features...")
    cfg = config['feature_engineering']
    df_out = df.copy()

    # Ensure Close exists for calculations
    if 'Close' not in df_out.columns:
        print("Warning: 'Close' column needed for enhanced features. Skipping some.")
        return df_out # Return original if Close is missing

    if cfg.get('add_volatility', False):
        df_out['log_ret'] = np.log(df_out['Close'] / df_out['Close'].shift(1))
        min_periods_vol = 5
        if len(df_out) > min_periods_vol:
             df_out['realized_vol_5'] = df_out['log_ret'].rolling(min_periods_vol, min_periods=min_periods_vol).std() * np.sqrt(252) # Annualized approx
        else: df_out['realized_vol_5'] = 0

    if cfg.get('add_range_atr', False):
        if 'High' in df_out.columns and 'Low' in df_out.columns:
            df_out['range'] = df_out['High'] - df_out['Low']
            # Use TA Lib's ATR calculation (more robust)
            atr = AverageTrueRange(
                high=df_out['High'],
                low=df_out['Low'],
                close=df_out['Close'], # Requires Close price
                window=14,
                fillna=True # Fill initial NaNs
            )
            df_out['atr_14'] = atr.average_true_range()
        else:
            print("Warning: High/Low columns missing, cannot calculate Range/ATR.")
            df_out['range'] = 0
            df_out['atr_14'] = 0

    if cfg.get('add_fft', False):
        close_fft = np.fft.fft(df_out['Close'].values)
        fft_df = pd.DataFrame(index=df_out.index)
        for k in cfg.get('fft_periods', [5, 10]):
            if k < len(close_fft):
                fft_df[f'fft_abs_{k}'] = np.abs(close_fft[k])
                fft_df[f'fft_angle_{k}'] = np.angle(close_fft[k])
            else: fft_df[f'fft_abs_{k}'] = 0; fft_df[f'fft_angle_{k}'] = 0
        df_out = pd.concat([df_out, fft_df], axis=1)

    return df_out

def add_fourier_features(df, config):
    """Adds Fourier features based on the DataFrame's DatetimeIndex."""
    print("Adding Fourier time features...")
    cfg = config['feature_engineering']
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: DataFrame index is not DatetimeIndex. Skipping Fourier features.")
        return df

    df_out = df.copy()
    periods = cfg.get('fourier_periods', [365.25, 30.5]) # Annual, Monthly
    order = cfg.get('fourier_order', 5)
    timestamps_s = df.index.view(np.int64) // 10**9 # Convert index to seconds since epoch

    for period_days in periods:
        period_seconds = period_days * 24 * 3600
        # Calculate frequencies based on the period
        frequencies = [(i * 2 * np.pi / period_seconds) for i in range(1, order + 1)]
        for i, freq in enumerate(frequencies):
            order_num = i + 1
            df_out[f'fourier_sin_{period_days:.1f}d_{order_num}'] = np.sin(freq * timestamps_s)
            df_out[f'fourier_cos_{period_days:.1f}d_{order_num}'] = np.cos(freq * timestamps_s)

    return df_out

def apply_feature_engineering(df, config):
    """Applies all configured feature engineering steps."""
    df_featured = df.copy()

    # 1. Apply enhanced features (Volatility, FFT, Range/ATR)
    df_featured = add_enhanced_features(df_featured, config)

    # 2. Apply Fourier Time Features
    if config['feature_engineering'].get('add_fourier_time', False):
        df_featured = add_fourier_features(df_featured, config)

    # 3. Apply standard TA features from 'ta' library
    if config['feature_engineering'].get('use_ta_library', True):
        print("Adding technical indicators using 'ta' library...")
        # Ensure required columns exist after previous steps
        required_ta_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in df_featured.columns for col in required_ta_cols):
            try:
                df_featured = add_all_ta_features(
                    df_featured, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
                )
            except Exception as e:
                print(f"Warning: TA features failed: {e}. Check columns.")
        else:
            print("Warning: Skipping TA features due to missing required columns (Open, High, Low, Close, Volume).")

    print(f"Shape after all features: {df_featured.shape}")

    # Final cleanup
    df_featured.dropna(axis=1, how='all', inplace=True)
    df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_featured.fillna(method='ffill', inplace=True)
    df_featured.fillna(method='bfill', inplace=True)
    df_featured.fillna(0, inplace=True) # Final catch-all

    return df_featured