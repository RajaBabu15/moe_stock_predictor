# moe_model/components.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, LSTM, GRU, Dense, Dropout, Input, Conv1D, Add,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    Softmax, Multiply, Concatenate, Flatten, Bidirectional, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import numpy as np # Needed for types

# === Expert Layers ===

class TransformerExpert(Layer):
    """Transformer Encoder Block as an Expert."""
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, dropout=0.1, l2_reg=0.0, name="transformer_expert", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model; self.num_heads = num_heads; self.ff_dim = ff_dim
        self.dropout_rate = dropout; self.l2_reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
        # Ensure key_dim is compatible with d_model and num_heads
        if d_model % num_heads != 0:
             raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu", kernel_regularizer=self.l2_reg),
            Dropout(dropout),
            Dense(d_model, kernel_regularizer=self.l2_reg)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6); self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout); self.dropout2 = Dropout(dropout)
        self.final_pool = GlobalAveragePooling1D(); self.final_dense = Dense(1, activation='linear', name="expert_output")
        self._built_input_shape = None # Track build shape

    def build(self, input_shape):
        if self._built_input_shape == input_shape: # Avoid rebuilding unnecessarily
            return
        if input_shape[-1] is None:
             raise ValueError("The last dimension of the input shape cannot be None.")
        # Project input features to d_model if they don't match
        self.input_proj = Dense(self.d_model, kernel_regularizer=self.l2_reg, name="input_projection") if input_shape[-1] != self.d_model else tf.identity
        super().build(input_shape)
        self._built_input_shape = input_shape # Mark as built

    def call(self, inputs, training=False):
        # Ensure build is called if needed (e.g., when loaded)
        if not self.built: self.build(inputs.shape)

        x = self.input_proj(inputs)
        attn_output = self.attn(query=x, value=x, key=x, training=training) # Use query, value, key explicitly
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1, training=training)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        pooled_output = self.final_pool(out2)
        return self.final_dense(pooled_output)

    def get_config(self):
        config = super().get_config()
        config.update({ "d_model": self.d_model, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "dropout": self.dropout_rate, "l2_reg": self.l2_reg.l2 if self.l2_reg else 0.0 }); return config


def build_tcn_expert(input_shape, config, name="tcn_expert", global_l2_reg=0.0):
    """Builds a Temporal Convolutional Network (TCN) expert Model."""
    filters = config.get("filters", 64); kernel_size = config.get("kernel_size", 3)
    dilations = config.get("dilations", [1, 2, 4, 8]); dropout = config.get("dropout", 0.1)
    # Use specific expert l2 if provided, else global
    l2_reg_val = config.get("l2_reg", global_l2_reg)
    l2_reg = regularizers.l2(l2_reg_val) if l2_reg_val > 0 else None
    mixed_precision = config.get("use_mixed_precision", False) # Need global config here? Pass it.

    inputs = Input(shape=input_shape, name=f"{name}_input"); x = inputs
    for dilation_rate in dilations:
        residual = Conv1D(filters, 1, padding='same', kernel_regularizer=l2_reg)(x) if x.shape[-1] != filters else x
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu', kernel_regularizer=l2_reg)(x)
        x = LayerNormalization()(x); x = Dropout(dropout)(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu', kernel_regularizer=l2_reg)(x)
        x = LayerNormalization()(x); x = Dropout(dropout)(x); x = Add()([x, residual])
    x = GlobalAveragePooling1D(name=f"{name}_pool")(x)
    outputs = Dense(1, activation='linear', name=f"{name}_output")(x)
    if mixed_precision: outputs = tf.keras.layers.Activation('linear', dtype='float32', name=f'{name}_output_float32')(outputs)
    return Model(inputs=inputs, outputs=outputs, name=name)

# Placeholder for NBEATS - requires a dedicated implementation library or code
# def build_nbeats_expert(...): ...

# === Gating Layers / Functions ===

def build_simple_gating(input_shape, config, gating_type="LSTM", name="simple_gating"):
    """Builds simpler LSTM or Dense based gating Model."""
    num_experts = config['num_experts']; l2_reg_val = config.get("l2_reg", 0.0)
    l2_reg = regularizers.l2(l2_reg_val) if l2_reg_val > 0 else None; gate_cfg = config['gating_config']
    dropout = gate_cfg.get("dropout", 0.2); inputs = Input(shape=input_shape, name=f"{name}_input"); x = inputs
    mixed_precision = config.get("use_mixed_precision", False)

    if gating_type == "LSTM":
        units_list = gate_cfg.get("lstm_units", [32])
        for i, units in enumerate(units_list):
             is_last = (i == len(units_list) - 1)
             layer = LSTM(units, return_sequences=not is_last, name=f"gate_lstm_{i+1}", kernel_regularizer=l2_reg)
             x = Bidirectional(layer) if gate_cfg.get("use_bidirectional", False) and i==0 else layer(x) # Optional Bidirectional on first layer
             x = Dropout(dropout, name=f"gate_drop_{i+1}")(x)
    elif gating_type == "Dense":
        units_list = gate_cfg.get("dense_units", [32]); x = Flatten(name="gate_flatten")(x)
        for i, units in enumerate(units_list):
            x = Dense(units, activation='relu', name=f"gate_dense_{i+1}", kernel_regularizer=l2_reg)(x)
            x = Dropout(dropout, name=f"gate_drop_{i+1}")(x)
    else: raise ValueError(f"Unknown simple gating type: {gating_type}")
    gate_logits = Dense(num_experts, name="gate_logits", kernel_regularizer=l2_reg)(x)
    gate_outputs = Softmax(name="gate_softmax_outputs")(gate_logits)
    if mixed_precision:
        gate_outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='gate_output_float32')(gate_logits)
        gate_logits = tf.cast(gate_logits, dtype='float32')
    return Model(inputs=inputs, outputs=[gate_outputs, gate_logits], name=name)

def context_aware_gating(inputs, config, name="context_aware_gating"):
    """Gating network using LSTM + Simple Attention + Feature Pooling Model."""
    gate_cfg = config['gating_config']; num_experts = config['num_experts']
    l2_reg_val = config.get("l2_reg", 0.0); l2_reg = regularizers.l2(l2_reg_val) if l2_reg_val > 0 else None
    dropout = gate_cfg.get("dropout", 0.2)
    mixed_precision = config.get("use_mixed_precision", False)
    contexts = []

    if gate_cfg.get("use_temporal_context", True):
        temporal = inputs
        for units in gate_cfg.get("lstm_units", [32]):
             layer = LSTM(units, return_sequences=True, kernel_regularizer=l2_reg)
             temporal = Bidirectional(layer) if gate_cfg.get("use_bidirectional", False) else layer(temporal)
             temporal = Dropout(dropout)(temporal)
        temporal_attn = Attention()([temporal, temporal])
        temporal_context = GlobalAveragePooling1D(name="gate_temporal_pool")(temporal_attn); contexts.append(temporal_context)

    if gate_cfg.get("use_feature_context", True):
        feature_context = GlobalAveragePooling1D(name="gate_feature_pool")(inputs)
        feature_context = Dense(gate_cfg.get("dense_units", [32])[0], activation='gelu', kernel_regularizer=l2_reg, name="gate_feature_dense")(feature_context)
        feature_context = Dropout(dropout)(feature_context); contexts.append(feature_context)

    if not contexts: raise ValueError("Context-aware gating requires at least one context type.")
    combined_context = Concatenate(name="gate_combined_context")(contexts) if len(contexts) > 1 else contexts[0]
    gate_logits = Dense(num_experts, name="gate_logits", kernel_regularizer=l2_reg)(combined_context)
    gate_outputs = Softmax(name="gate_softmax_outputs")(gate_logits)
    if mixed_precision:
        gate_outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='gate_output_float32')(gate_logits)
        gate_logits = tf.cast(gate_logits, dtype='float32')
    return Model(inputs=inputs, outputs=[gate_outputs, gate_logits], name=name)


def attention_mha_gating(inputs, config, name="attention_mha_gating"):
    """Gating based on BiLSTM + MultiHeadAttention Model."""
    gate_cfg = config['gating_config']
    num_experts = config['num_experts']
    l2_reg_val = config.get("l2_reg", 0.0)
    l2_reg = regularizers.l2(l2_reg_val) if l2_reg_val > 0 else None
    dropout = gate_cfg.get("dropout", 0.2)
    lstm_units = gate_cfg.get("lstm_units", [32])[0]
    num_heads = gate_cfg.get("mha_heads", 4)
    key_dim = gate_cfg.get("mha_key_dim", 32)
    use_bidirectional = gate_cfg.get("use_bidirectional", True)
    mixed_precision = config.get("use_mixed_precision", False)

    # Create model
    input_layer = Input(shape=inputs.shape[1:], name=f"{name}_input")
    x = input_layer

    # LSTM layer
    rnn_layer = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2_reg)
    if use_bidirectional:
        x = Bidirectional(rnn_layer)(x)
    else:
        x = rnn_layer(x)
    x = Dropout(dropout)(x)

    # Ensure key_dim compatibility
    effective_dim = lstm_units * 2 if use_bidirectional else lstm_units
    if key_dim * num_heads != effective_dim:
        print(f"Warning: MHA input dim ({effective_dim}) != heads*key_dim ({num_heads}*{key_dim}). Adjusting key_dim.")
        if effective_dim % num_heads == 0:
            key_dim = effective_dim // num_heads
            print(f"Adjusted key_dim to {key_dim}")
        else:
            x = Dense(num_heads * key_dim, activation='relu')(x)  # Project to compatible dimension

    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(query=x, value=x, key=x)
    x = GlobalAveragePooling1D()(attn_output)
    gate_logits = Dense(num_experts, name="gate_logits", kernel_regularizer=l2_reg)(x)
    gate_outputs = Softmax(name="gate_softmax_outputs")(gate_logits)

    if mixed_precision:
        gate_outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='gate_output_float32')(gate_logits)
        gate_logits = tf.cast(gate_logits, dtype='float32')

    return Model(inputs=input_layer, outputs=[gate_outputs, gate_logits], name=name)


# Placeholder for CrossAttention Gating - needs careful thought on how to integrate
# def cross_attention_gating(expert_outputs, inputs, config): ...


# === Loss Functions ===

class LoadBalancingLossLayer(tf.keras.layers.Layer):
    def __init__(self, num_experts, **kwargs):
        super(LoadBalancingLossLayer, self).__init__(**kwargs)
        self.num_experts = num_experts

    def call(self, gate_logits):
        # Cast inputs to float32
        gate_logits = tf.cast(gate_logits, tf.float32)
        
        # Calculate softmax of gate logits
        gates = tf.nn.softmax(gate_logits, axis=-1)
        
        # Calculate the fraction of tokens routed to each expert
        # Shape: [num_experts]
        router_prob = tf.reduce_mean(gates, axis=0)
        
        # Calculate the auxiliary load balancing loss
        # We want a uniform distribution of 1/num_experts for each expert
        uniform_prob = tf.ones_like(router_prob) / tf.cast(self.num_experts, tf.float32)
        aux_loss = tf.reduce_sum(router_prob * tf.math.log(router_prob / uniform_prob))
        
        # Add the auxiliary loss to the model's losses
        self.add_loss(aux_loss)
        
        return aux_loss

def load_balancing_loss(gate_logits, num_experts):
    """
    Compute load balancing loss for mixture of experts gating.
    Now returns a Keras Layer that can be used in the model.
    
    Args:
        gate_logits: Tensor of shape [batch_size, num_experts] containing gate logits
        num_experts: Number of experts in the mixture
    
    Returns:
        A LoadBalancingLossLayer instance
    """
    return LoadBalancingLossLayer(num_experts)(gate_logits)


def directional_loss(y_true, y_pred, weight=0.1):
    """Improved directional loss comparing price *changes*."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate MSE on the original values
    mse_loss = K.mean(K.square(y_true - y_pred))

    # Calculate differences (price changes) - requires at least 2 points
    # This works if y_true/y_pred are shaped (batch, seq_len_out) where seq_len_out >= 2
    # If shape is (batch, 1) - predicting single next value:
    # We cannot calculate change directly from output. Need previous value.
    # Heuristic: Compare sign of predicted value vs sign of true value (less meaningful for price)
    # Option: Predict return instead of price if using this loss.

    # Assuming output shape allows difference calculation (e.g., predicting >1 step):
    if tf.shape(y_true)[-1] > 1:
        true_diff = y_true[..., 1:] - y_true[..., :-1]
        pred_diff = y_pred[..., 1:] - y_pred[..., :-1]

        true_direction = K.sign(true_diff)
        pred_direction = K.sign(pred_diff)

        # Penalize when directions mismatch
        direction_penalty = K.mean(K.relu(1.0 - true_direction * pred_direction))
    else:
        # Cannot calculate direction from single value output, return only MSE
        print("Warning: Directional loss cannot calculate direction from single output value. Returning MSE only.")
        direction_penalty = tf.constant(0.0, dtype=tf.float32)


    return mse_loss + direction_penalty * weight

class GateWeightLogger(tf.keras.callbacks.Callback):
    """Callback to log gate weights during training."""
    def __init__(self, log_dir='logs/gate_weights', log_freq=5):
        super().__init__()
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_freq == 0:
            # Get the gate weights from the model
            gate_layer = None
            for layer in self.model.layers:
                if 'gate_softmax_outputs' in layer.name:
                    gate_layer = layer
                    break
            
            if gate_layer is not None:
                with self.writer.as_default():
                    # Log the gate weights distribution
                    tf.summary.histogram('gate_weights', gate_layer.weights[0], step=epoch)
                    self.writer.flush()

    def on_train_end(self, logs=None):
        if hasattr(self, 'writer'):
            self.writer.close()