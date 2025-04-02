import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, LayerNormalization, Dropout, Add, GlobalAveragePooling1D, Dense

@tf.keras.utils.register_keras_serializable(package="moe_model")
class TCNExpert(keras.layers.Layer):
    def __init__(self, nb_filters=64, kernel_size=3, nb_stacks=3, dilations=None, dropout_rate=0.1, **kwargs):
        super(TCNExpert, self).__init__(**kwargs)
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dilations = dilations if dilations is not None else [1, 2, 4]
        self.dropout_rate = dropout_rate
        
        self.conv_layers = []
        self.norm_layers = []
        self.dropout_layers = []
        self.residual_layers = []
        
        for s in range(nb_stacks):
            for d in self.dilations:
                self.conv_layers.append(Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                            dilation_rate=d, padding='same', activation='relu'))
                self.norm_layers.append(LayerNormalization(epsilon=1e-6))
                self.dropout_layers.append(Dropout(dropout_rate))
                if s > 0:  # Add residual connections after first stack
                    self.residual_layers.append(Add())
        
        self.global_pool = GlobalAveragePooling1D()
        self.final_layer = Dense(1)

    def call(self, inputs, training=None):
        x = inputs
        skip_connections = []
        
        for i in range(len(self.conv_layers)):
            res = x
            x = self.conv_layers[i](x)
            x = self.norm_layers[i](x)
            x = self.dropout_layers[i](x, training=training)
            
            if i >= len(self.dilations):  # After first stack
                x = self.residual_layers[i - len(self.dilations)]([x, res])
            
            skip_connections.append(x)
        
        x = self.global_pool(x)
        output = self.final_layer(x)
        
        return output

    def get_config(self):
        config = super(TCNExpert, self).get_config()
        config.update({
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'nb_stacks': self.nb_stacks,
            'dilations': self.dilations,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config) 