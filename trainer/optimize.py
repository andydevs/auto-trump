"""
Find optimal model dimensions using scikit optimize
"""
from .model import create_model
from .data import input_data
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras import backend as K
from argparse import ArgumentParser
import json

# Argument Parser
parser = ArgumentParser(description='Optimize neural network')
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
    default=5,
    help='number of epochs to train for')
parser.add_argument('--ncalls',
    dest='ncalls',
    type=int,
    default=20,
    help='number of optimization iterations')
args = parser.parse_args()

# Optimal parameters save location
PARAM_SAVE_LOCATION = 'files/support/hyperparams.json'

# Dimensions
dim_embedding_units = Integer(low=100, high=750, name='embedding_units')
dim_lstm_units = Integer(low=100, high=750, name='lstm_units')
dim_dense_units = Integer(low=100, high=750, name='dense_units')
dim_dropout_rate = Real(low=0.0, high=0.25, name='dropout_rate')
dimensions = [ dim_embedding_units, dim_lstm_units, 
    dim_dense_units, dim_dropout_rate ]

# Load current accuracy and parameters
print('Loading current hyperparameters...')
best_accuracy = 0.0
current_hyper_parameters = [ 250, 500, 750, 0.2 ]
with open(PARAM_SAVE_LOCATION, 'r') as f:
    params = json.load(f)
    best_accuracy = params['_accuracy']
    current_hyper_parameters = [
        params['embedding_units'],
        params['lstm_units'],
        params['dense_units'],
        params['dropout_rate']
    ]

# Get dataset
train_data, test_data, vocab_size = input_data(
    display_data=False,
    train_frac=args.train_frac,
    batch=args.batch,
    repeat=args.repeat,
    shuffle=args.shuffle)

@use_named_args(dimensions=dimensions)
def optimize_model_fun(embedding_units, lstm_units, dense_units, dropout_rate):
    """
    Find optimal model in search space
    """
    global best_accuracy
    print('----------------------------------------------------------------------')
    print('Dropout Rate:', dropout_rate)

    # Create, train, and evaluate model
    model = create_model(vocab_size, embedding_units, lstm_units, dense_units, dropout_rate)
    model.fit(train_data, epochs=args.epochs, callbacks=[ReduceLROnPlateau(monitor='loss')])
    loss, accuracy = model.evaluate(test_data)
    print(f'Accuracy: {accuracy:0.2%}')

    # Delete model and clear session
    # this is really important!
    del model
    K.clear_session()

    # Save params if accuracy improved
    if accuracy > best_accuracy:
        print('Saving new params...')
        with open(PARAM_SAVE_LOCATION, 'w') as f:
            json.dump({
                'embedding_units': int(embedding_units),
                'lstm_units': int(lstm_units),
                'dense_units': int(dense_units),
                'dropout_rate': float(dropout_rate),
                '_accuracy': float(accuracy)
            }, f, indent=4)
        best_accuracy = accuracy

    # Return accuracy (negative because we're minimizing)
    return -accuracy


# Run optimization
results = gp_minimize(
    func=optimize_model_fun,
    dimensions=dimensions,
    acq_func='EI',
    n_calls=args.ncalls,
    x0=current_hyper_parameters)