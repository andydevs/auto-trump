"""
Run training task
"""
from argparse import ArgumentParser
from .data import input_data
from .model import create_model_for_vocab_size

# Saved model file
MODEL_FILE = 'files/models/saved-model.h5'


def train_and_evaluate_model(dataset, num_words, embedding_dims, lstm_units, epochs):
    """
    Train and evaluate model
    """
    model = create_model_for_vocab_size(num_words, embedding_dims, lstm_units)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
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
    parser.add_argument('--embedding-dims',
        dest='embedding_dims',
        type=int,
        default=6,
        help='Number of output dimensions in embedding layer')
    parser.add_argument('--lstm-units',
        dest='embedding_dims',
        type=int,
        default=24,
        help='Number of output units in lstm layer')
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

    # Retrieve data
    dataset, num_words = input_data(
        display_data=args.display_data,
        batch=args.batch,
        repeat=args.repeat,
        shuffle=args.shuffle)

    # Run training task
    if args.train:
        train_and_evaluate_model(
            dataset=dataset,
            num_words=num_words,
            embedding_dims=args.embedding_dims,
            lstm_units=args.lstm_units,
            epochs=args.epochs)