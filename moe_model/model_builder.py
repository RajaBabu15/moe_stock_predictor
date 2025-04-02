# moe_model/model_builder.py
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, GRU, Input, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Concatenate, Lambda, Multiply, Softmax
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# Import necessary components from within the package
from .components import (TransformerExpert, build_tcn_expert,
                         build_simple_gating, context_aware_gating, attention_mha_gating,
                         load_balancing_loss, directional_loss)
# Import LSTM/GRU from Keras layers directly if needed in fallback
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

# QAT import moved here, only if needed
try:
    import tensorflow_model_optimization as tfmot
    HAS_TF_OPTIMIZATION = True
except ImportError:
    HAS_TF_OPTIMIZATION = False
    print("Warning: tensorflow_model_optimization not found. Quantization Aware Training disabled.")

@tf.keras.utils.register_keras_serializable(package="moe_model", name="reshape_expert_outputs")
def reshape_expert_outputs(expert_outputs):
    return tf.expand_dims(tf.expand_dims(expert_outputs, axis=1), axis=-1)

@tf.keras.utils.register_keras_serializable(package="moe_model")
def reshape_gate_outputs(gate_outputs):
    """Reshapes gate outputs for weighted sum aggregation."""
    return tf.expand_dims(tf.expand_dims(gate_outputs, axis=1), axis=-1)

@tf.keras.utils.register_keras_serializable(package="moe_model")
def sum_weighted_outputs(weighted_experts):
    """Sums the weighted expert outputs."""
    return tf.squeeze(tf.reduce_sum(weighted_experts, axis=2), axis=[1, -1])

def build_expert_network_dispatch(input_shape, expert_config, global_config, name_prefix="expert"):
    """Dispatcher to build different expert types using components."""
    expert_type = expert_config.get("type", "LSTM")
    l2_reg_val = global_config.get("l2_reg", 0.0) # Use global L2 by default
    name = f"{name_prefix}_{expert_type}"
    mixed_precision = global_config.get("use_mixed_precision", False)

    if expert_type == "LSTM" or expert_type == "GRU":
        units_list = expert_config.get("units", [64, 32])
        dropout = expert_config.get("dropout", 0.2)
        l2_reg = tf.keras.regularizers.l2(l2_reg_val) if l2_reg_val > 0 else None
        inputs = Input(shape=input_shape, name=f"{name}_input"); x = inputs
        for i, units in enumerate(units_list):
            is_last_rnn_layer = (i == len(units_list) - 1); return_sequences = not is_last_rnn_layer
            LayerClass = LSTM if expert_type == "LSTM" else GRU
            x = LayerClass(units=units, return_sequences=return_sequences, name=f"{name}_{i+1}", kernel_regularizer=l2_reg)(x)
            x = Dropout(dropout, name=f"{name}_drop_{i+1}")(x)
        outputs = Dense(1, activation='linear', name=f"{name}_output")(x)
        model = Model(inputs=inputs, outputs=outputs, name=name)

    elif expert_type == "Transformer":
        transformer_config = {k: expert_config.get(k) for k in ["d_model", "num_heads", "ff_dim", "dropout"]}
        transformer_config['l2_reg'] = l2_reg_val
        inputs = Input(shape=input_shape, name=f"{name}_input")
        expert_layer = TransformerExpert(**transformer_config, name=name)
        outputs = expert_layer(inputs)
        model = Model(inputs=inputs, outputs=outputs, name=name)

    elif expert_type == "TCN":
        # Pass expert_config and global L2
        model = build_tcn_expert(input_shape, expert_config, name=name, global_l2_reg=l2_reg_val)

    # Placeholder for NBEATS
    elif expert_type == "NBEATS":
         print(f"Warning: Expert type '{expert_type}' not fully implemented. Needs specific library/code.")
         # Fallback or raise error
         raise NotImplementedError("NBEATS expert needs implementation.")

    else: raise ValueError(f"Unsupported expert_type: {expert_type}")

    # Apply final output casting if needed (TCN/Transformer handle internally)
    if expert_type in ["LSTM", "GRU"] and mixed_precision:
         outputs = tf.keras.layers.Activation('linear', dtype='float32', name=f'{name}_output_float32')(model.output)
         model = Model(inputs=model.inputs, outputs=outputs, name=model.name)

    return model


def build_moe_model(input_shape, config):
    """Builds the complete MoE model."""
    main_input = Input(shape=input_shape, name="main_input", dtype=tf.float32)
    num_experts = config["num_experts"]
    l2_reg = config.get("l2_reg", 0.0)

    # 1. Build Gating Network
    gating_type = config['gating_config'].get('type', 'AttentionMHA')
    print(f"Building Gating Network (Type: {gating_type})...")
    if gating_type == "ContextAware":
        gating_network = context_aware_gating(main_input, config)
    elif gating_type == "AttentionMHA":
        gating_network = attention_mha_gating(main_input, config)
    elif gating_type == "SimpleLSTM":
        gating_network = build_simple_gating(input_shape, config, "LSTM")
    elif gating_type == "SimpleDense":
        gating_network = build_simple_gating(input_shape, config, "Dense")
    else:
        raise ValueError(f"Unsupported gating type: {gating_type}")
    
    gate_outputs, gate_logits = gating_network(main_input)

    # 2. Build Expert Networks
    print("Building Expert Networks...")
    expert_outputs = []
    for i, expert_conf in enumerate(config["expert_configs"]):
        print(f" - Building Expert {i+1} (Type: {expert_conf.get('type', 'LSTM')})")
        expert_model = build_expert_network_dispatch(input_shape, expert_conf, config, name_prefix=f"expert_{i}")
        expert_output = expert_model(main_input)
        expert_outputs.append(expert_output)

    # 3. Combine Experts
    aggregation_method = config.get("aggregation", "WeightedSum")
    print(f"Combining experts using: {aggregation_method}")
    if aggregation_method == "WeightedSum":
        # Convert list of expert outputs to a single tensor using Concatenate
        stacked_expert_outputs = Concatenate(axis=-1, name="expert_concat")(expert_outputs)
        # Reshape expert outputs to match gate outputs
        expert_reshape = Lambda(reshape_expert_outputs, output_shape=(None, 1, num_experts, 1))(stacked_expert_outputs)
        # Reshape gate outputs to match expert outputs
        gate_reshape = Lambda(reshape_gate_outputs, output_shape=(None, 1, num_experts, 1))(gate_outputs)
        # Multiply expert outputs with gate outputs
        weighted_experts = Multiply()([expert_reshape, gate_reshape])
        # Sum the weighted expert outputs
        final_output = Lambda(sum_weighted_outputs, output_shape=(None,))(weighted_experts)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

    # Add load balancing loss if specified
    if config.get('use_load_balancing', False):
        lb_loss_value = load_balancing_loss(gate_logits, num_experts)
        
    # Create the base model
    base_model = tf.keras.Model(
        inputs=main_input,
        outputs=final_output,
        name="moe_model"
    )

    # Compile the model with the specified optimizer and loss
    optimizer = tf.keras.optimizers.get(config.get('optimizer', 'adam'))
    loss_fn = tf.keras.losses.get(config.get('loss', 'mean_squared_error'))
    
    # Enable mixed precision if specified
    if config.get('use_mixed_precision', False):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    base_model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['mae', 'mse']  # Add any additional metrics here
    )

    return base_model