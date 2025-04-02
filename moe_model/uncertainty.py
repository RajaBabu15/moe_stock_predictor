# moe_model/uncertainty.py
import tensorflow as tf
from tensorflow.keras.models import Model

class ProbabilisticMoE(Model):
    """Wraps a model to perform vectorized MC Dropout during prediction."""
    def __init__(self, base_model, num_samples=50, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.num_samples = num_samples
        self.supports_masking = True # Indicate masking support if base_model does

    def call(self, inputs, training=False, mask=None):
        if not training:
            # Vectorized MC sampling
            # 1. Tile inputs num_samples times along a new batch dimension
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            num_features = tf.shape(inputs)[2]
            # Tile inputs: shape becomes (batch_size * num_samples, seq_len, num_features)
            tiled_inputs = tf.repeat(inputs, repeats=self.num_samples, axis=0)

            # 2. Run inference with training=True to enable dropout
            # Base model should handle the larger batch size
            mc_predictions = self.base_model(tiled_inputs, training=True, mask=mask)

            # 3. Reshape back to (num_samples, batch_size, output_dim)
            output_shape = tf.shape(mc_predictions)[1:] # Get shape after sequence/feature dims
            # New shape: (num_samples, batch_size, ...)
            reshaped_output = tf.reshape(mc_predictions, (self.num_samples, batch_size, -1)) # Flatten output dims if needed
            # If output is just (batch, 1), reshape to (num_samples, batch_size, 1)
            if len(output_shape) == 0 or output_shape[0] == 1:
                 final_shape = (self.num_samples, batch_size, 1)
            else: # Handle potentially multi-output models? Assume (batch, output_dim) -> (num_samples, batch_size, output_dim)
                 final_shape = tf.concat([[self.num_samples], [batch_size], output_shape], axis=0)

            return tf.reshape(mc_predictions, final_shape)
        else:
            # Pass through during actual training
            return self.base_model(inputs, training=True, mask=mask)

    # Delegate saving/loading to the base model if needed, although saving the wrapper might be complex
    # It's often better to save the base_model and wrap it after loading for prediction.
    def get_config(self):
        # Configuration to recreate the wrapper (not the base model itself)
        config = super().get_config()
        config.update({
            "base_model": tf.keras.utils.serialize_keras_object(self.base_model), # Serialize base model reference
            "num_samples": self.num_samples,
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_model_config = config.pop("base_model")
        base_model = tf.keras.utils.deserialize_keras_object(base_model_config)
        return cls(base_model, **config)