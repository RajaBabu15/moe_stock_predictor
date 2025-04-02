# tune.py
import keras_tuner as kt
import tensorflow as tf
import os
import joblib

# Import project modules
from config.config import CONFIG as BASE_CONFIG # Import base config
from moe_model.data_loader import load_and_prepare_data
from moe_model.feature_engineering import apply_feature_engineering
from moe_model.preprocessing import scale_and_sequence_data, create_tf_dataset
from moe_model.model_builder import build_moe_model
from tensorflow.keras.callbacks import EarlyStopping

# --- Tuner Configuration ---
TUNER_CONFIG = {
    "project_name": "moe_stock_hyper_tuning",
    "objective": "val_loss",
    "max_trials": 20,
    "executions_per_trial": 1,
    "directory": "hyperparam_tuning",
}

class MoEHyperModel(kt.HyperModel):
    def __init__(self, base_config, input_shape):
        self.base_config = base_config
        self.input_shape = input_shape

    def build(self, hp):
        """Builds the MoE model with hyperparameters."""
        # Create a mutable copy of the config for this trial
        trial_config = self.base_config.copy()
        trial_config['gating_config'] = self.base_config['gating_config'].copy()
        # Ensure expert_configs is deep copied if modifying structure
        trial_config['expert_configs'] = [cfg.copy() for cfg in self.base_config['expert_configs']]

        # --- Define Hyperparameters ---
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 5e-4, 1e-4])
        hp_l2_reg = hp.Float("l2_reg", min_value=1e-5, max_value=1e-2, sampling="log")
        hp_dropout = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
        # Example: Tune number of experts (requires adjusting expert_configs list length)
        # hp_num_experts = hp.Int("num_experts", min_value=2, max_value=5, step=1)
        # Example: Tune LSTM units in first expert
        hp_lstm_units_1 = hp.Int("lstm_units_1", min_value=32, max_value=128, step=32)
        hp_lstm_units_2 = hp.Int("lstm_units_2", min_value=16, max_value=64, step=16)

        # --- Update Trial Config ---
        trial_config["learning_rate"] = hp_learning_rate
        trial_config["l2_reg"] = hp_l2_reg
        # Apply dropout globally or per component if desired
        trial_config["gating_config"]["dropout"] = hp_dropout
        for cfg in trial_config["expert_configs"]:
             cfg["dropout"] = hp_dropout # Apply same dropout to all experts for simplicity here

        # Update specific expert params (example for first LSTM expert)
        if trial_config["expert_configs"][0]["type"] == "LSTM":
             trial_config["expert_configs"][0]["units"] = [hp_lstm_units_1, hp_lstm_units_2]

        # --- Build Model ---
        # Note: build_moe_model should use the trial_config
        model = build_moe_model(self.input_shape, trial_config)
        return model # Return the compiled model

if __name__ == "__main__":
    print("--- Starting Hyperparameter Tuning ---")

    # 1. Load and Prepare Data (once)
    try:
        df, target_col_name = load_and_prepare_data(BASE_CONFIG)
        df_featured = apply_feature_engineering(df, BASE_CONFIG)
        X_train, y_train, X_test, y_test, _, num_features, _ = scale_and_sequence_data(
            df_featured, target_col_name, BASE_CONFIG
        )
    except Exception as e: print(f"Fatal Error during data processing: {e}"); exit()
    if X_train.shape[0] < BASE_CONFIG["batch_size"] or X_test.shape[0] == 0: print("Error: Insufficient data."); exit()

    train_dataset = create_tf_dataset(X_train, y_train, BASE_CONFIG["batch_size"], shuffle=True)
    validation_dataset = create_tf_dataset(X_test, y_test, BASE_CONFIG["batch_size"])
    if train_dataset is None or validation_dataset is None: print("Error: Failed to create TF Datasets."); exit()

    input_shape_for_tuner = (BASE_CONFIG["sequence_length"], num_features)

    # 2. Initialize Tuner
    hypermodel = MoEHyperModel(base_config=BASE_CONFIG, input_shape=input_shape_for_tuner)

    # Choose a tuner (RandomSearch, Hyperband, BayesianOptimization)
    tuner = kt.RandomSearch(
        hypermodel,
        objective=TUNER_CONFIG["objective"],
        max_trials=TUNER_CONFIG["max_trials"],
        executions_per_trial=TUNER_CONFIG["executions_per_trial"],
        directory=TUNER_CONFIG["directory"],
        project_name=TUNER_CONFIG["project_name"],
        overwrite=True # Overwrite previous tuning results
    )

    tuner.search_space_summary()

    # 3. Run Search
    print("\n--- Running Hyperparameter Search ---")
    stop_early = EarlyStopping(monitor='val_loss', patience=10) # Shorter patience for tuning
    tuner.search(
        train_dataset,
        epochs=BASE_CONFIG["epochs"] // 2, # Use fewer epochs for faster tuning
        validation_data=validation_dataset,
        callbacks=[stop_early],
        verbose=1 # Set to 2 for less output per epoch
    )

    # 4. Get Best Hyperparameters and Model
    print("\n--- Tuning Complete ---")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    Best Hyperparameters Found:
    Learning Rate: {best_hps.get('learning_rate')}
    L2 Regularization: {best_hps.get('l2_reg'):.6f}
    Dropout Rate: {best_hps.get('dropout_rate'):.2f}
    LSTM Units 1: {best_hps.get('lstm_units_1')}
    LSTM Units 2: {best_hps.get('lstm_units_2')}
    """) # Add other tuned HPs here

    # Build the best model
    best_model = tuner.hypermodel.build(best_hps)

    # Optional: Retrain the best model with full epochs on the complete dataset
    print("\n--- Retraining Best Model ---")
    history = best_model.fit(
        train_dataset,
        epochs=BASE_CONFIG["epochs"], # Full epochs
        validation_data=validation_dataset,
        callbacks=[stop_early], # Can use early stopping again
        verbose=1
    )

    # Save the final best model
    best_model_path = os.path.join(TUNER_CONFIG["directory"], "best_moe_model_tuned.keras")
    best_model.save(best_model_path)
    print(f"Best tuned model saved to: {best_model_path}")

    print("\n--- Hyperparameter Tuning Finished ---")