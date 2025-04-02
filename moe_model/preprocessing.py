# moe_model/preprocessing.py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def create_sequences(features, target, seq_length):
    """Creates sequences with the target shifted correctly."""
    xs, ys = [], []
    if features.shape[0] != target.shape[0]:
        raise ValueError(f"Features length ({features.shape[0]}) and target length ({target.shape[0]}) must match.")
    if len(features) <= seq_length:
         print(f"Warning: Data length ({len(features)}) is not greater than sequence length ({seq_length}). No sequences created.")
         return np.array([]).reshape(0, seq_length, features.shape[1] if features.ndim > 1 else 1), np.array([])

    for i in range(len(features) - seq_length):
        x = features[i:(i + seq_length)]
        y = target[i + seq_length] # Predict the value *after* the sequence ends
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def scale_and_sequence_data(df, target_col_name, config):
    """Scales data, creates sequences, and splits into train/test."""
    sequence_length = config["sequence_length"]
    split_ratio = config["train_test_split_ratio"]

    # Select features (exclude original target and the 'Target' column)
    feature_cols = [col for col in df.columns if col not in [target_col_name, 'Target']]
    if not feature_cols:
        raise ValueError("No feature columns found after processing.")

    # Extract target based on original column name *before* it was shifted
    if target_col_name not in df.columns:
        raise ValueError(f"Original target column '{target_col_name}' not found for scaling.")
    target_raw = df[target_col_name].values

    features_df = df[feature_cols]
    num_features = features_df.shape[1]
    print(f"Number of features being used for scaling: {num_features}")

    # --- Scaling ---
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features_df)

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target_raw = scaler_target.fit_transform(target_raw.reshape(-1, 1)).flatten()

    # --- Splitting Data Chronologically ---
    split_index = int(len(scaled_features) * split_ratio)
    if split_index < sequence_length or (len(scaled_features) - split_index) < sequence_length:
         print("Warning: Split results in partitions too small for sequence length.")
         # Handle this case: maybe adjust split or raise error? For now, proceed with caution.

    train_features = scaled_features[:split_index]
    test_features = scaled_features[split_index:]
    train_target_raw = scaled_target_raw[:split_index]
    test_target_raw = scaled_target_raw[split_index:]

    # --- Creating Sequences ---
    X_train, y_train = create_sequences(train_features, train_target_raw, sequence_length)
    X_test, y_test = create_sequences(test_features, test_target_raw, sequence_length)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
         print("Warning: No sequences created for train or test set. Check data length and sequence_length.")

    return X_train, y_train, X_test, y_test, scaler_target, num_features, scaler_features

def create_tf_dataset(X, y, batch_size, shuffle=False, buffer_size=1000):
    """Creates a tf.data.Dataset pipeline."""
    if X is None or y is None or X.shape[0] == 0:
        print("Warning: Cannot create TF Dataset from empty or None arrays.")
        return None
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X shape {X.shape} and y shape {y.shape} mismatch for TF Dataset.")

    # Ensure data types are float32 for TensorFlow compatibility
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset