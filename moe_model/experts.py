import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import Dense, LSTM, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D

@register_keras_serializable()
class TransformerExpert(tf.keras.layers.Layer):
    def __init__(self, d_model=64, num_heads=4, dff=128, dropout_rate=0.1, **kwargs):
        super(TransformerExpert, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # Multi-head attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        # Layer normalization and dropout
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Global average pooling and output
        self.global_pool = layers.GlobalAveragePooling1D()
        self.output_layer = layers.Dense(1)
        
    def call(self, inputs, training=False):
        # Self-attention
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        # Global pooling and output
        pooled = self.global_pool(out2)
        return self.output_layer(pooled)
    
    def get_config(self):
        config = super(TransformerExpert, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config 