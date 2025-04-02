# tests/test_preprocessing.py
import unittest
import numpy as np
import pandas as pd

# Import the function to test (adjust path if needed)
from moe_model.preprocessing import create_sequences

class TestPreprocessing(unittest.TestCase):

    def test_sequence_creation_basic(self):
        """Test sequence creation with simple data."""
        features = np.arange(20).reshape(10, 2) # 10 steps, 2 features
        target = np.arange(10) * 10            # 10 target values
        seq_length = 3

        X, y = create_sequences(features, target, seq_length)

        self.assertEqual(X.shape, (7, 3, 2)) # 10 - 3 = 7 sequences
        self.assertEqual(y.shape, (7,))
        # Check first sequence target: should be target[0 + 3] = target[3] = 30
        self.assertEqual(y[0], 30)
        # Check last sequence target: should be target[6 + 3] = target[9] = 90
        self.assertEqual(y[-1], 90)
        # Check first sequence features
        np.testing.assert_array_equal(X[0], features[0:3])
        # Check last sequence features
        np.testing.assert_array_equal(X[-1], features[6:9]) # indices 6, 7, 8

    def test_sequence_creation_edge_case_short(self):
        """Test sequence creation when data length equals seq_length."""
        features = np.random.rand(5, 3)
        target = np.random.rand(5)
        seq_length = 5
        X, y = create_sequences(features, target, seq_length)
        self.assertEqual(X.shape[0], 0) # Expect 0 sequences
        self.assertEqual(y.shape[0], 0)

    def test_sequence_creation_edge_case_too_short(self):
        """Test sequence creation when data length < seq_length."""
        features = np.random.rand(4, 3)
        target = np.random.rand(4)
        seq_length = 5
        X, y = create_sequences(features, target, seq_length)
        self.assertEqual(X.shape[0], 0) # Expect 0 sequences
        self.assertEqual(y.shape[0], 0)

    def test_sequence_creation_mismatch_length(self):
        """Test error handling for mismatched feature/target lengths."""
        features = np.random.rand(10, 2)
        target = np.random.rand(9) # Mismatched length
        seq_length = 3
        with self.assertRaises(ValueError):
            create_sequences(features, target, seq_length)

if __name__ == '__main__':
    unittest.main()