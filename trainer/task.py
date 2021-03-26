"""
Run training task
"""
import tensorflow as tf
from argparse import ArgumentParser
from .data import input_data

# Saved model file
MODEL_FILE = 'files/models/saved-model.h5'


def train_and_evaluate_model(dataset, vocab_size, train, epochs):
    """
    Train and evaluate model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 100),
        tf.keras.layers.LSTM(750, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(750),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy', 'mse'])
    model.summary()
    if train:
        model.fit(dataset, epochs=epochs)
        model.save(MODEL_FILE)


if __name__ == '__main__':
    # Parse args
    parser = ArgumentParser(description='Run training task')
    parser.add_argument('--no-train',
        dest='train',
        action='store_false',
        help="don't train model. Just run data step")
    parser.add_argument('--display-data',
        dest='display_data',
        action='store_true',
        help='display sample of data after preprocessing.')
    parser.add_argument('--batch',
        dest='batch',
        type=int,
        default=200,
        help='number of datapoints in a training batch')
    parser.add_argument('--repeat',
        dest='repeat',
        type=int,
        default=5,
        help='number of times the dataset is repeated')
    parser.add_argument('--shuffle',
        dest='shuffle',
        type=int,
        default=200,
        help='size of shuffle buffer')
    parser.add_argument('--epochs',
        dest='epochs',
        type=int,
        default=10,
        help='number of epochs to train for')
    args = parser.parse_args()

    # Retrieve data and run training task
    dataset, vocab_size = input_data(
        display_data=args.display_data,
        batch=args.batch,
        repeat=args.repeat,
        shuffle=args.shuffle)
    train_and_evaluate_model(
        dataset=dataset,
        vocab_size=vocab_size,
        train=args.train,
        epochs=args.epochs)
