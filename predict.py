# predict.py
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib # For loading scalers
import os

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

# Import project modules
from config.config import CONFIG # Use config for defaults like seq length
from moe_model.components import TransformerExpert # Need custom layer for loading
from moe_model.uncertainty import ProbabilisticMoE
# Assume data loading/feature/preprocessing functions can be reused or simplified for prediction
from moe_model.data_loader import load_and_prepare_data # Simplified use
from moe_model.feature_engineering import apply_feature_engineering
from moe_model.preprocessing import scale_and_sequence_data # Only need sequence part

# Add imports for the Lambda functions
from moe_model.model_builder import (
    reshape_expert_outputs,
    reshape_gate_outputs,
    sum_weighted_outputs
)

def load_model_for_prediction(config):
    """Loads the trained Keras model."""
    checkpoint_path = os.path.join(config["checkpoint_dir"], config["checkpoint_filename"])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Trained model not found at {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")
    # Update custom_objects to include Lambda functions
    custom_objects = {
        'TransformerExpert': TransformerExpert,
        'moe_model>reshape_expert_outputs': reshape_expert_outputs,
        'moe_model>reshape_gate_outputs': reshape_gate_outputs,
        'moe_model>sum_weighted_outputs': sum_weighted_outputs
    }
    try:
        model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def prepare_prediction_data(config):
    """Loads recent data and prepares the last sequence for prediction."""
    # 1. Load data (enough for sequence length + features)
    # Simplified loading, assuming file contains recent data
    try:
        # Load more data than just sequence length to allow for feature calculation lookback
        # Adjust this logic based on actual prediction needs (e.g., load last N days)
        df_raw, target_col_name = load_and_prepare_data(config) # Loads entire file for simplicity here
        df_featured = apply_feature_engineering(df_raw, config)
    except Exception as e:
        print(f"Error preparing prediction data: {e}")
        return None, None, None, None

    # 2. Get feature columns (must match training)
    feature_cols = [col for col in df_featured.columns if col not in [target_col_name, 'Target']]
    if not feature_cols:
        print("Error: No feature columns found in prediction data.")
        return None, None, None, None

    # 3. Load scalers (assuming they were saved during training)
    # Placeholder paths - replace with actual saved paths
    scaler_features_path = 'scaler_features.joblib'
    scaler_target_path = 'scaler_target.joblib'
    if os.path.exists(scaler_features_path) and os.path.exists(scaler_target_path):
         scaler_features = joblib.load(scaler_features_path)
         scaler_target = joblib.load(scaler_target_path)
    else:
         print("Error: Scaler files not found. Cannot proceed with prediction.")
         # As a fallback for demo: refit scalers on available data (NOT recommended for production)
         print("Warning: Refitting scalers on available data - results may be inaccurate.")
         scaler_features = MinMaxScaler(feature_range=(0, 1)).fit(df_featured[feature_cols])
         scaler_target = MinMaxScaler(feature_range=(0, 1)).fit(df_featured[[target_col_name]])
         # Consider saving them now if needed:
         # joblib.dump(scaler_features, scaler_features_path)
         # joblib.dump(scaler_target, scaler_target_path)


    # 4. Scale the features
    scaled_features = scaler_features.transform(df_featured[feature_cols])

    # 5. Extract the last sequence
    sequence_length = config["sequence_length"]
    if len(scaled_features) < sequence_length:
        print(f"Error: Not enough data ({len(scaled_features)}) to form a sequence of length {sequence_length}.")
        return None, None, None, None

    last_sequence_scaled = scaled_features[-sequence_length:]
    # Reshape for model input (batch_size=1, seq_len, num_features)
    last_sequence_scaled = np.expand_dims(last_sequence_scaled, axis=0).astype(np.float32)

    # Get the last known actual price for context (optional)
    last_actual_price = df_raw[target_col_name].iloc[-1]


    return last_sequence_scaled, scaler_target, last_actual_price, target_col_name


if __name__ == "__main__":
    print("--- Making Prediction ---")

    # 1. Load Trained Model
    try:
        trained_base_model = load_model_for_prediction(CONFIG)
    except Exception as e:
        print(f"Failed to load model. Exiting. Error: {e}")
        exit()

    # 2. Prepare Input Data (Last Sequence)
    input_sequence, scaler_target, last_actual, target_name = prepare_prediction_data(CONFIG)

    if input_sequence is None:
        print("Failed to prepare prediction data. Exiting.")
        exit()

    # 3. Make Prediction
    if CONFIG.get("predict_uncertainty", False):
        print("Performing MC Dropout prediction (50 samples)...")
        mc_samples_scaled = []
        for _ in range(50):
            pred = trained_base_model(input_sequence, training=True)
            if hasattr(pred, 'numpy'):
                mc_samples_scaled.append(pred.numpy())
            else:
                mc_samples_scaled.append(pred)
        mc_samples_scaled = np.array(mc_samples_scaled)
        pred_mean_scaled = np.mean(mc_samples_scaled)
        pred_std_scaled = np.std(mc_samples_scaled)

        pred_mean_unscaled = scaler_target.inverse_transform([[pred_mean_scaled]])[0, 0]
        # Approximate unscaled std dev
        pred_std_unscaled = pred_std_scaled * (scaler_target.data_max_[0] - scaler_target.data_min_[0])

        print(f"\nPrediction for next step ({target_name}):")
        print(f"  Mean: {pred_mean_unscaled:.4f}")
        print(f"  Std Dev (Uncertainty): {pred_std_unscaled:.4f}")
        print(f"  95% CI Approx: [{pred_mean_unscaled - 1.96 * pred_std_unscaled:.4f}, {pred_mean_unscaled + 1.96 * pred_std_unscaled:.4f}]")

    else:
        print("Performing standard prediction...")
        prediction_scaled = trained_base_model(input_sequence) # Shape (1, 1)
        prediction_unscaled = scaler_target.inverse_transform(prediction_scaled)[0, 0]
        print(f"\nPrediction for next step ({target_name}): {prediction_unscaled:.4f}")

    print(f"(Last known actual price: {last_actual:.4f})")

    print("\n--- Prediction Finished ---")