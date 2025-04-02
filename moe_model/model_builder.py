# moe_model/model_builder.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Multiply, Softmax, Add, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
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
    _tfmot_found = True
except ImportError:
    _tfmot_found = False
    print("Warning: tensorflow_model_optimization not found. Quantization Aware Training disabled.")


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
    main_input = Input(shape=input_shape, name="main_input", dtype=tf.float32) # Ensure float32 input
    num_experts = config["num_experts"]
    l2_reg = config.get("l2_reg", 0.0)

    # 1. Build Gating Network
    gating_type = config['gating_config'].get('type', 'AttentionMHA')
    print(f"Building Gating Network (Type: {gating_type})...")
    if gating_type == "ContextAware": gating_network = context_aware_gating(main_input, config)
    elif gating_type == "AttentionMHA": gating_network = attention_mha_gating(main_input, config)
    elif gating_type == "SimpleLSTM": gating_network = build_simple_gating(input_shape, config, "LSTM")
    elif gating_type == "SimpleDense": gating_network = build_simple_gating(input_shape, config, "Dense")
    # Placeholder for CrossAttention - needs expert outputs, different flow
    # elif gating_type == "CrossAttention": ...
    else: raise ValueError(f"Unsupported gating type: {gating_type}")
    gate_outputs, gate_logits = gating_network(main_input) # gate_outputs shape: (batch, num_experts)

    # 2. Build Expert Networks
    print("Building Expert Networks...")
    expert_outputs = []
    for i, expert_conf in enumerate(config["expert_configs"]):
        print(f" - Building Expert {i+1} (Type: {expert_conf.get('type', 'LSTM')})")
        # Use the dispatcher function
        expert_model = build_expert_network_dispatch(input_shape, expert_conf, config, name_prefix=f"expert_{i}")
        expert_output = expert_model(main_input) # Get output tensor
        expert_outputs.append(expert_output)

    # 3. Combine Experts
    aggregation_method = config.get("aggregation", "WeightedSum")
    print(f"Combining experts using: {aggregation_method}")
    if aggregation_method == "WeightedSum":
        stacked_expert_outputs = tf.stack(expert_outputs, axis=1) # (batch, num_experts, 1)
        # Ensure gate_outputs is float32 for multiplication, esp. with mixed precision
        gate_outputs_f32 = tf.cast(gate_outputs, tf.float32)
        expanded_gate_outputs = tf.expand_dims(gate_outputs_f32, axis=-1) # (batch, num_experts, 1)
        # Ensure expert outputs are float32 before multiplication
        stacked_expert_outputs_f32 = tf.cast(stacked_expert_outputs, tf.float32)
        weighted_expert_outputs = Multiply(name="weighted_experts")([stacked_expert_outputs_f32, expanded_gate_outputs])
        final_output = tf.reduce_sum(weighted_expert_outputs, axis=1, name="final_output") # (batch, 1)
    else: raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

    # Create the base Keras model
    base_model = Model(inputs=main_input, outputs=final_output, name="moe_stock_predictor_base")

    # 4. Optional: Apply Quantization Aware Training
    if config.get("quantization_aware_training", False):
        if _tfmot_found:
            print("Applying Quantization Aware Training wrapper...")
            base_model = tfmot.quantization.keras.quantize_model(base_model)
            print("QAT wrapper applied.")
        else:
            print("Warning: Quantization Aware Training requested but tensorflow_model_optimization not found. Skipping.")


    # 5. Compile the Model (with auxiliary losses handled correctly)
    optimizer = Adam(learning_rate=config["learning_rate"])
    if config.get("use_mixed_precision", False):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # Define primary loss function
    loss_choice = config.get("loss_function", "mean_squared_error")
    if loss_choice == "directional_loss":
        primary_loss_fn = lambda yt, yp: directional_loss(yt, yp, config.get("directional_loss_weight", 0.1))
        print(f"Using Directional Loss (Weight: {config.get('directional_loss_weight', 0.1)}) as primary loss.")
    else:
        primary_loss_fn = 'mean_squared_error'
        print("Using Mean Squared Error as primary loss.")

    # Compile the model with the primary loss
    base_model.compile(optimizer=optimizer, loss=primary_loss_fn, metrics=['mae'])
    print("Model compiled with primary loss and metrics.")

    # Add auxiliary losses *after* compiling with primary loss
    if config.get("use_load_balancing_loss", False):
        # Need access to gate_logits from the gating_network Model inside base_model
        # Find the gating network layer/model
        gating_layer = None
        for layer in base_model.layers:
            if isinstance(layer, Model) and 'gating' in layer.name:
                 gating_layer = layer
                 break
        if gating_layer:
             # Use the logits output (index 1) of the gating layer
             logits_tensor = gating_layer.outputs[1]
             lb_loss_value = load_balancing_loss(logits_tensor, num_experts)
             base_model.add_loss(config["load_balancing_loss_weight"] * lb_loss_value)
             # Add metric manually if needed (compile doesn't automatically add aux loss metrics)
             base_model.add_metric(lb_loss_value, name='load_balancing_loss', aggregation='mean')
             print(f"Load Balancing auxiliary loss added (weight: {config['load_balancing_loss_weight']}).")
        else:
             print("Warning: Could not find gating network layer to add load balancing loss.")


    return base_model