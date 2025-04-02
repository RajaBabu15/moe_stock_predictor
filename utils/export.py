# utils/export.py
import tensorflow as tf
import os

# Defer ONNX imports until function call to avoid mandatory dependency
def export_to_onnx(model, input_shape, output_path):
    """Exports a Keras model to ONNX format."""
    try:
        import onnx
        import tf2onnx
    except ImportError:
        print("Error: 'onnx' and 'tf2onnx' packages are required for ONNX export.")
        print("Install them using: pip install onnx tf2onnx")
        return

    print(f"Exporting model to ONNX format at: {output_path}")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Define the input signature - use specific batch size None for flexibility
    spec = (tf.TensorSpec((None,) + input_shape, tf.float32, name="input"),)

    try:
        # Convert the model
        # opset=13 is a common target, adjust if needed
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        # Save the ONNX model
        onnx.save(model_proto, output_path)
        print("Model exported successfully to ONNX.")
    except Exception as e:
        print(f"Error during ONNX export: {e}")