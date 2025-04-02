# moe_model/uncertainty.py
import tensorflow as tf
from tensorflow.keras.layers import Layer

class ProbabilisticMoE(Layer):
    """Wraps a model to perform vectorized MC Dropout during prediction."""
    def __init__(self, base_model, num_samples=10, **kwargs):
        super(ProbabilisticMoE, self).__init__(**kwargs)
        self.base_model = base_model
        self.num_samples = num_samples
        self.supports_masking = True # Indicate masking support if base_model does

    def call(self, inputs, training=None):
        if training:
            # During training, just return the base model's prediction and a placeholder for variance
            predictions = self.base_model(inputs, training=training)
            return predictions, tf.zeros_like(predictions)
        else:
            # During inference, use MC Dropout
            mc_samples = []
            for _ in range(self.num_samples):
                predictions = self.base_model(inputs, training=True)  # Enable dropout during inference
                mc_samples.append(predictions)
            
            # Stack samples and compute statistics
            mc_samples = tf.stack(mc_samples, axis=0)  # Shape: [num_samples, batch_size, 1]
            mean = tf.reduce_mean(mc_samples, axis=0)
            variance = tf.math.reduce_variance(mc_samples, axis=0)
            return mean, variance

    def predict(self, inputs, batch_size=None):
        # Implement predict method for compatibility
        mean, variance = self(inputs, training=False)
        return {'mean': mean, 'variance': variance}

    # Delegate saving/loading to the base model if needed, although saving the wrapper might be complex
    # It's often better to save the base_model and wrap it after loading for prediction.
    def get_config(self):
        # Configuration to recreate the wrapper (not the base model itself)
        config = super().get_config()
        config.update({
            "base_model": self.base_model,
            "num_samples": self.num_samples,
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_model = config.pop("base_model")
        return cls(base_model, **config)