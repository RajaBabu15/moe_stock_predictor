# app.py
import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# Import necessary project components
from config.config import CONFIG
from moe_model.components import TransformerExpert # For loading model
from moe_model.uncertainty import ProbabilisticMoE
# Need simplified versions or reuse parts of data pipeline for Gradio input
from moe_model.feature_engineering import apply_feature_engineering # Simplified use
from moe_model.preprocessing import scale_and_sequence_data # Simplified use

# --- Global Variables (Load Model and Scalers once) ---
MODEL = None
SCALER_FEATURES = None
SCALER_TARGET = None
TARGET_COL_NAME = CONFIG["target_column"] # Assume config is correct
FEATURE_COLS = None # Determined after loading data first time

def load_resources():
    """Loads model and scalers."""
    global MODEL, SCALER_FEATURES, SCALER_TARGET, FEATURE_COLS, TARGET_COL_NAME

    if MODEL is not None: # Avoid reloading
        return True

    # Load Model
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_filename"])
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        return False
    try:
        custom_objects = {'TransformerExpert': TransformerExpert}
        MODEL = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects, compile=False)
        print("Model loaded for Gradio app.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False

    # Load Scalers (Requires saving them during training)
    scaler_features_path = 'scaler_features.joblib'
    scaler_target_path = 'scaler_target.joblib'
    if os.path.exists(scaler_features_path) and os.path.exists(scaler_target_path):
        SCALER_FEATURES = joblib.load(scaler_features_path)
        SCALER_TARGET = joblib.load(scaler_target_path)
        # Determine feature columns from scaler
        if hasattr(SCALER_FEATURES, 'feature_names_in_'):
             FEATURE_COLS = SCALER_FEATURES.feature_names_in_
        else: # Fallback - try to infer (less reliable)
             print("Warning: Could not get feature names from scaler. Prediction might fail.")
        print("Scalers loaded.")
        return True
    else:
        print(f"ERROR: Scaler files ({scaler_features_path}, {scaler_target_path}) not found. Please train the model and save scalers.")
        return False


def predict_stock(input_df):
    """Takes DataFrame input, preprocesses, predicts, and returns prediction."""
    if not load_resources():
        return "Error: Model or Scalers not loaded.", None

    if not isinstance(input_df, pd.DataFrame):
         # Gradio might pass dict, convert if needed
         try: input_df = pd.DataFrame(input_df)
         except: return "Error: Invalid input format. Expected DataFrame.", None

    print(f"Received input data with {len(input_df)} rows.")
    if len(input_df) < CONFIG["sequence_length"]:
        return f"Error: Input data must have at least {CONFIG['sequence_length']} rows.", None

    # Ensure necessary columns are present
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] # Base required
    if not all(col in input_df.columns for col in required_cols):
         return f"Error: Input DataFrame missing required columns: {required_cols}", None

    # Preprocessing similar to prediction script
    try:
        input_df['Date'] = pd.to_datetime(input_df['Date'])
        input_df = input_df.set_index('Date')

        # Ensure target column exists for feature engineering steps if needed
        if TARGET_COL_NAME not in input_df.columns:
             if 'Close' in input_df.columns: TARGET_COL_NAME = 'Close'
             else: return "Error: Target column ('Close' or configured) missing.", None

        df_featured = apply_feature_engineering(input_df, CONFIG)

        # Get features matching the scaler
        if FEATURE_COLS is None: # If feature names weren't loaded
             current_feature_cols = [col for col in df_featured.columns if col not in [TARGET_COL_NAME, 'Target']] # Recalculate
        else:
             current_feature_cols = [col for col in FEATURE_COLS if col in df_featured.columns] # Use known feature names

        if not current_feature_cols: return "Error: No matching feature columns found after processing.", None

        # Keep only the required feature columns
        df_final_features = df_featured[current_feature_cols]

        scaled_features = SCALER_FEATURES.transform(df_final_features)
        last_sequence_scaled = scaled_features[-CONFIG["sequence_length"]:]
        input_sequence = np.expand_dims(last_sequence_scaled, axis=0).astype(np.float32)

    except Exception as e:
        return f"Error during preprocessing: {e}", None

    # Prediction
    try:
        if CONFIG.get("predict_uncertainty", False):
            prob_model = ProbabilisticMoE(MODEL, num_samples=CONFIG['mc_dropout_samples'])
            mc_samples_scaled = prob_model.predict(input_sequence)
            mc_samples_scaled = tf.squeeze(mc_samples_scaled).numpy()
            pred_mean_scaled = np.mean(mc_samples_scaled)
            pred_std_scaled = np.std(mc_samples_scaled)
            pred_mean = SCALER_TARGET.inverse_transform([[pred_mean_scaled]])[0, 0]
            pred_std = pred_std_scaled * (SCALER_TARGET.data_max_[0] - SCALER_TARGET.data_min_[0])
            result_text = f"Predicted Price: {pred_mean:.2f} (+/- {1.96 * pred_std:.2f} 95% CI)"
        else:
            prediction_scaled = MODEL.predict(input_sequence)
            pred_mean = SCALER_TARGET.inverse_transform(prediction_scaled)[0, 0]
            result_text = f"Predicted Price: {pred_mean:.2f}"

        # Create a simple plot dataframe (e.g., last N actual + prediction)
        plot_df = input_df[[TARGET_COL_NAME]].tail(30).reset_index() # Last 30 actual points
        # Add predicted point (need to create a future date)
        last_date = plot_df['Date'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1) # Simple next day assumption
        pred_row = pd.DataFrame({'Date': [next_date], TARGET_COL_NAME: [pred_mean]})
        plot_df = pd.concat([plot_df, pred_row], ignore_index=True)


        return result_text, plot_df

    except Exception as e:
        return f"Error during prediction: {e}", None


# --- Create Gradio Interface ---
# Define inputs: Use file upload or direct dataframe editor
# Input example: Upload CSV with Date, Open, High, Low, Close, Volume columns
# Ensure Date is parsed correctly by pandas later

# Simplest Input: DataFrame editor (might be slow/clunky for large data)
# Create sample input dataframe structure
sample_input_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
sample_data = {col: [100.0 + i for i in range(CONFIG['sequence_length'])] for col in sample_input_cols if col != 'Date'}
sample_data['Date'] = pd.date_range(end=datetime.date.today(), periods=CONFIG['sequence_length'], freq='B').strftime('%Y-%m-%d')
sample_df = pd.DataFrame(sample_data)

input_component = gr.DataFrame(
    value=sample_df, # Provide initial sample data
    label="Input Stock Data (requires Date, Open, High, Low, Close, Volume columns)",
    headers=sample_input_cols,
    row_count=(CONFIG['sequence_length'], "fixed"), # Fixed row count needed
    col_count=(len(sample_input_cols), "fixed")
    )

output_text = gr.Text(label="Prediction Result")
# Output plot needs Date and the Target Column name
output_plot = gr.LinePlot(x="Date", y=TARGET_COL_NAME, label="Price Trend + Prediction", width=700, height=400)


interface = gr.Interface(
    fn=predict_stock,
    inputs=input_component,
    outputs=[output_text, output_plot],
    title="MoE Stock Price Predictor",
    description=f"Upload or edit the DataFrame with the last {CONFIG['sequence_length']} days of stock data (including today) to predict the next day's {TARGET_COL_NAME} price.",
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Launching Gradio Dashboard...")
    # Load resources once on startup if possible (might have issues with multiprocessing)
    # load_resources()
    interface.launch() # share=True for public link