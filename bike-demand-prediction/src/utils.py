"""
Utility functions
"""
import numpy as np
import tensorflow as tf


def set_seed(seed_value=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)