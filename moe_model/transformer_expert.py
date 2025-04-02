import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D

@tf.keras.utils.register_keras_serializable(package="moe_model")
class TransformerExpert(keras.layers.Layer):
    def __init__(self, d_model=64, num_heads=4, dff=128, dropout_rate=0.1, **kwargs):
        super(TransformerExpert, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn_layer1 = Dense(dff, activation='relu')
        self.ffn_layer2 = Dense(d_model)
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
        self.global_pool = GlobalAveragePooling1D()
        self.final_layer = Dense(1)

    def call(self, inputs, training=None):
        # Self attention
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn_layer1(out1)
        ffn_output = self.ffn_layer2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        # Global pooling and final prediction
        pooled = self.global_pool(out2)
        output = self.final_layer(pooled)
        
        return output

    def get_config(self):
        config = super(TransformerExpert, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config) 