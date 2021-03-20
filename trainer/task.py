"""
Run training task
"""
from argparse import ArgumentParser
from .data import input_data
from .model import create_model_for_vocab_size

# Saved model file
MODEL_FILE = 'files/models/saved-model.h5'


def train_and_evaluate_model(dataset, num_words, epochs):
    """
    Train and evaluate model
    """
    model = create_model_for_vocab_size(num_words)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    model.fit(dataset, epochs=epochs)
    model.save(MODEL_FILE)


if __name__ == '__main__':
    # Parse args
    parser = ArgumentParser(description='Run training task')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.add_argument('--display-data', dest='display_data', action='store_true')
    parser.add_argument('--batch', dest='batch', type=int, default=20)
    parser.add_argument('--repeat', dest='repeat', type=int, default=5)
    parser.add_argument('--shuffle', dest='shuffle', type=int, default=200)
    parser.add_argument('--epochs', dest='epochs', type=int, default=5)
    args = parser.parse_args()

    # Retrieve data
    dataset, num_words = input_data(
        display_data=args.display_data,
        batch=args.batch,
        repeat=args.repeat,
        shuffle=args.shuffle)
    if args.train:
        train_and_evaluate_model(
            dataset=dataset,
            num_words=num_words,
            epochs=args.epochs)