"""
Define model
"""
import tensorflow as tf

def create_model_for_vocab_size(vocab_size, embedding_dims, lstm_units):
    """
    Create model given vocabulary size
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dims),
        tf.keras.layers.LSTM(lstm_units),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])