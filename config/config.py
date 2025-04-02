# config/config.py
"""Central configuration for the MoE Stock Predictor project."""

CONFIG = {
    # --- Data ---
    "csv_file": "data/sample_data.csv",
    "target_column": "Close",
    "sequence_length": 60,
    "train_test_split_ratio": 0.8,

    # --- Feature Engineering ---
    "feature_engineering": {
        "use_ta_library": True,
        "add_volatility": True,
        "add_range_atr": True, # Use TA Lib ATR calculation
        "add_fft": True,
        "fft_periods": [5, 10, 20],
        "add_fourier_time": True, # Add Fourier time features
        "fourier_periods": [365.25, 30.5], # Annual, Monthly approx
        "fourier_order": 5,
    },

    # --- MoE Architecture ---
    "num_experts": 3, # Will be overridden by len(expert_configs)
    "expert_configs": [ # Define configurations for each expert
        {"type": "LSTM", "units": [64, 32], "dropout": 0.2},
        {"type": "Transformer", "d_model": 64, "num_heads": 4, "ff_dim": 128, "dropout": 0.1},
        {"type": "TCN", "filters": 64, "kernel_size": 3, "dilations": [1, 2, 4], "dropout": 0.1},
        # {"type": "NBEATS", "stack_type": "trend", ...} # Placeholder - requires implementation
    ],
    "gating_config": {
        "type": "AttentionMHA", # 'SimpleLSTM', 'SimpleDense', 'ContextAware', 'AttentionMHA', 'CrossAttention'
        "lstm_units": [32],
        "dense_units": [32],
        "mha_heads": 4,
        "mha_key_dim": 32,
        "use_bidirectional": True, # For AttentionMHA
        "use_temporal_context": True, # For ContextAware
        "use_feature_context": True, # For ContextAware
        "dropout": 0.2
    },
    "aggregation": "WeightedSum",

    # --- Training & Optimization ---
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.001,
    "l2_reg": 0.0001,
    "loss_function": "mean_squared_error", # 'mean_squared_error', 'directional_loss'
    "directional_loss_weight": 0.1, # Weight if directional_loss is used
    "use_mixed_precision": False,
    "use_load_balancing_loss": True,
    "load_balancing_loss_weight": 0.01,
    "quantization_aware_training": False, # Enable QAT

    # --- Callbacks / Saving ---
    "use_early_stopping": True,
    "early_stopping_patience": 15,
    "use_model_checkpointing": True,
    "checkpoint_dir": "model_checkpoints",
    "checkpoint_filename": "best_moe_model.keras",
    "use_tensorboard": True,
    "tensorboard_log_dir": "logs/fit",
    "log_gate_weights": True,
    "gate_log_freq": 5,

    # --- Uncertainty & Evaluation ---
    "predict_uncertainty": True,
    "mc_dropout_samples": 50,
    "calculate_financial_metrics": True,

    # --- Deployment ---
    "onnx_export_path": "model_onnx/moe_predictor.onnx",
}

# Auto-adjust num_experts based on expert_configs length
CONFIG["num_experts"] = len(CONFIG["expert_configs"])