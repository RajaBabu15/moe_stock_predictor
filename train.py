# train.py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Import project modules
from config.config import CONFIG
from moe_model.data_loader import load_and_prepare_data
from moe_model.feature_engineering import apply_feature_engineering
from moe_model.preprocessing import scale_and_sequence_data, create_tf_dataset
from moe_model.model_builder import build_moe_model
from moe_model.uncertainty import ProbabilisticMoE
from moe_model.components import GateWeightLogger # Import custom callback
from utils.plotting import plot_training_history, plot_predictions
from utils.metrics import calculate_financial_metrics
from utils.export import export_to_onnx

def train_and_evaluate(config):
    """Loads data, builds, trains, evaluates, and plots the MoE model."""
    print("\n--- 1. Data Loading & Preparation ---")
    try:
        df, target_col_name = load_and_prepare_data(config)
        df_featured = apply_feature_engineering(df, config)
        X_train, y_train, X_test, y_test, scaler_target, num_features, _ = scale_and_sequence_data(
            df_featured, target_col_name, config
        )
    except Exception as e:
        print(f"Fatal Error during data processing: {e}"); return None, None
    if X_train.shape[0] < config["batch_size"] or X_test.shape[0] == 0:
        print("Error: Insufficient data after processing."); return None, None

    print("\n--- 2. Creating TF Datasets ---")
    train_dataset = create_tf_dataset(X_train, y_train, config["batch_size"], shuffle=True)
    # Use test set for validation data
    validation_dataset = create_tf_dataset(X_test, y_test, config["batch_size"])
    if train_dataset is None or validation_dataset is None:
        print("Error: Failed to create TF Datasets."); return None, None

    print("\n--- 3. Building MoE Model ---")
    input_shape = (config["sequence_length"], num_features)
    base_model = build_moe_model(input_shape, config)
    base_model.summary(expand_nested=True)

    print("\n--- 4. Training Model ---")
    # --- Callbacks ---
    callbacks_list = []
    if config["use_early_stopping"]:
        callbacks_list.append(EarlyStopping(monitor='val_loss', patience=config["early_stopping_patience"], restore_best_weights=True, verbose=1))

    checkpoint_path = os.path.join(config["checkpoint_dir"], config["checkpoint_filename"])
    if config["use_model_checkpointing"]:
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        # Save weights only can be safer if custom objects cause issues loading full model
        callbacks_list.append(ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1))

    ts_log_dir = None
    if config["use_tensorboard"]:
        ts_log_dir = os.path.join(config["tensorboard_log_dir"], datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{base_model.name}")
        callbacks_list.append(TensorBoard(log_dir=ts_log_dir, histogram_freq=1))
        print(f"TensorBoard logs: {ts_log_dir}")

    if config["log_gate_weights"]:
        # Ensure log dir exists for gate logger if TensorBoard is off
        gate_tb_dir = ts_log_dir if ts_log_dir else os.path.join("logs", "gate_weights")
        os.makedirs(gate_tb_dir, exist_ok=True)
        callbacks_list.append(GateWeightLogger(log_freq=config["gate_log_freq"], log_dir=gate_tb_dir))

    # --- Fit ---
    start_time = time.time()
    history = base_model.fit(
        train_dataset,
        epochs=config["epochs"],
        validation_data=validation_dataset,
        callbacks=callbacks_list,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time:.2f} seconds.")

    # --- Load Best Model (Optional but recommended) ---
    if config["use_model_checkpointing"] and os.path.exists(checkpoint_path):
        print(f"Loading best model weights from: {checkpoint_path}")
        try:
            # Load the full model structure and weights
            # Define custom objects needed for loading layers like TransformerExpert
            custom_objects = {'TransformerExpert': TransformerExpert} # Add others if needed
            # Load model - re-compilation might be needed if optimizer state isn't saved correctly or loss is lambda
            loaded_model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects, compile=False)

            # Re-compile the loaded model (important if loss is custom lambda or optimizer state needed)
            optimizer = Adam(learning_rate=config["learning_rate"])
            if config.get("use_mixed_precision", False): optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            loss_choice = config.get("loss_function", "mean_squared_error")
            primary_loss_fn = (lambda yt, yp: directional_loss(yt, yp, config.get("directional_loss_weight", 0.1))) if loss_choice == "directional_loss" else 'mean_squared_error'
            loaded_model.compile(optimizer=optimizer, loss=primary_loss_fn, metrics=['mae'])

            # Add auxiliary losses again to the loaded model if they were used during training
            # This logic needs access to the internal gating layer's logits output
            if config.get("use_load_balancing_loss", False):
                gating_layer = None
                for layer in loaded_model.layers: # Find gating layer in loaded model
                    if isinstance(layer, Model) and 'gating' in layer.name: gating_layer = layer; break
                if gating_layer and len(gating_layer.outputs) > 1:
                    logits_tensor = gating_layer.outputs[1]
                    lb_loss_value = load_balancing_loss(logits_tensor, config["num_experts"])
                    loaded_model.add_loss(config["load_balancing_loss_weight"] * lb_loss_value)
                    loaded_model.add_metric(lb_loss_value, name='load_balancing_loss', aggregation='mean')
                    print("Re-added load balancing loss to loaded model.")
                else: print("Warning: Could not re-add load balancing loss to loaded model.")

            base_model = loaded_model # Use the loaded best model
            print("Best model loaded successfully.")

        except Exception as e:
            print(f"Warning: Could not load best model from checkpoint. Using model from last epoch. Error: {e}")


    print("\n--- 5. Evaluation ---")
    results = base_model.evaluate(validation_dataset, verbose=1)
    print(f"Evaluation Results - Metrics: {base_model.metrics_names}")
    print(f"Evaluation Results - Values: {[f'{v:.6f}' for v in results]}")
    test_mae_scaled = results[base_model.metrics_names.index('mae')] if 'mae' in base_model.metrics_names else float('nan')

    print("\n--- 6. Predictions & Uncertainty ---")
    y_test_unscaled = scaler_target.inverse_transform(y_test.reshape(-1, 1).astype(np.float32))

    if config["predict_uncertainty"]:
        prob_model = ProbabilisticMoE(base_model, num_samples=config['mc_dropout_samples'])
        print("Generating MC Dropout predictions...")
        # Predict on numpy array for simplicity with the wrapper
        mc_samples_scaled = prob_model.predict(X_test, batch_size=config['batch_size'])
        # Output shape: (num_samples, num_points, 1) -> Squeeze last dim
        mc_samples_scaled = tf.squeeze(mc_samples_scaled, axis=-1).numpy() # (num_samples, num_points)

        y_pred_mean_scaled = np.mean(mc_samples_scaled, axis=0) # (num_points,)
        y_pred_std_scaled = np.std(mc_samples_scaled, axis=0) # (num_points,)

        y_pred_mean = scaler_target.inverse_transform(y_pred_mean_scaled.reshape(-1, 1).astype(np.float32))
        y_pred_std = y_pred_std_scaled * (scaler_target.data_max_[0] - scaler_target.data_min_[0])
        print("MC Dropout finished.")
    else:
        print("Generating standard predictions...")
        y_pred_mean_scaled = base_model.predict(X_test, batch_size=config['batch_size'])
        y_pred_mean = scaler_target.inverse_transform(y_pred_mean_scaled.astype(np.float32))
        y_pred_std = None

    mae_unscaled = np.mean(np.abs(y_pred_mean - y_test_unscaled))
    print(f"Test MAE (Unscaled Price): {mae_unscaled:.4f} (Scaled MAE: {test_mae_scaled:.6f})")

    if config["calculate_financial_metrics"]:
         _ = calculate_financial_metrics(y_test_unscaled.flatten(), y_pred_mean.flatten())

    print("\n--- 7. Plotting Results ---")
    plot_training_history(history, config)
    plot_predictions(y_test_unscaled, y_pred_mean, y_pred_std, config, target_col_name)

    # 8. Optional: Export to ONNX
    if config.get("onnx_export_path"):
         export_to_onnx(base_model, input_shape, config["onnx_export_path"])


    return base_model, history

# === Main Execution ===
if __name__ == "__main__":
    # Apply mixed precision globally if configured
    if CONFIG.get("use_mixed_precision", False):
        print("Setting mixed precision policy.")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    trained_model, training_history = train_and_evaluate(CONFIG)

    if trained_model:
         print("\n--- Training Completed Successfully ---")
         # You can add further steps here, like saving scalers, etc.
         # joblib.dump(scaler_target, 'scaler_target.joblib')
         # joblib.dump(scaler_features, 'scaler_features.joblib')
    else:
         print("\n--- Training Failed ---")

    print("\n--- Script Finished ---")