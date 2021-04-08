"""
Run training task
"""
import tensorflow as tf
from argparse import ArgumentParser
from .data import input_data
from .model import create_model
from .ifttt import IFTTTTrainingProgressCallback, IFTTTTrainingCompleteCallback
import json

# Job Name
JOB_NAME = 'auto-trump'

# Saved model file
PARAM_SAVE_LOCATION = 'files/support/hyperparams.json'
MODEL_FILE = 'files/models/saved-model.h5'


def train_and_evaluate_model(train_dataset, test_dataset, model, epochs):
    """
    Train and evaluate model
    """
    model.fit(train_dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss'),
            IFTTTTrainingProgressCallback(JOB_NAME, epochs),
            IFTTTTrainingCompleteCallback(JOB_NAME)
        ])
    model.evaluate(test_dataset)
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
    parser.add_argument('--train-frac',
        dest='train_frac',
        type=float,
        default=0.75,
        help='Fraction of dataset reserved for training')
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

    # Retrieve dataset
    print('Generating dataset...')
    train_dataset, test_dataset, vocab_size = input_data(
        display_data=args.display_data,
        train_frac=args.train_frac,
        batch=args.batch,
        repeat=args.repeat,
        shuffle=args.shuffle)

    # Build model with hyperparams
    print('Loading params...')
    with open(PARAM_SAVE_LOCATION, 'r') as f:
        params = json.load(f)
        model = create_model(vocab_size, 
            embedding_units=params['embedding_units'],
            lstm_units=params['lstm_units'], 
            dense_units=params['dense_units'],
            dropout_rate=params['dropout_rate'])

    # Train model if training
    if args.train:
        print('Training...')
        train_and_evaluate_model(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            epochs=args.epochs)
