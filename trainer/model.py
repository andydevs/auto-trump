"""
Define model function
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Global dropout rate
DROPOUT_RATE = 0.2

def create_model(vocab_size, embedding_units, lstm_units, dense_units):
    """
    Create text prediction model
    """
    model = Sequential([
        Embedding(vocab_size, embedding_units),
        Dropout(DROPOUT_RATE),
        LSTM(lstm_units, recurrent_dropout=DROPOUT_RATE),
        Dropout(DROPOUT_RATE),
        Dense(dense_units, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    return model