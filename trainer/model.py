"""
Define model
"""
import tensorflow as tf

def create_model_for_vocab_size(vocab_size):
    """
    Create model given vocabulary size
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 6),
        tf.keras.layers.LSTM(24),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])