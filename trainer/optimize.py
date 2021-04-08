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
import json
import tensorflow as tf

# Optimal parameters save location
OPTIMAL_SAVE_LOCATION = 'files/support/optimal-params.json'

# Set logging
tf.get_logger().setLevel('INFO')

# Dimensions
dim_embedding_units = Integer(low=100, high=1000, name='embedding_units')
dim_lstm_units = Integer(low=100, high=1000, name='lstm_units')
dim_dense_units = Integer(low=100, high=1000, name='dense_units')
dim_dropout_rate = Real(low=0.0, high=0.25, name='dropout_rate')
dimensions = [ dim_embedding_units, dim_lstm_units, 
    dim_dense_units, dim_dropout_rate ]

# Load current optimal parameters
print('Loading current optimal parameters...')
current_optimal_parameters = [ 250, 500, 750, 0.2 ]
with open(OPTIMAL_SAVE_LOCATION, 'r') as f:
    params = json.load(f)
    current_optimal_parameters = [
        params['embedding_units'],
        params['lstm_units'],
        params['dense_units'],
        params['dropout_rate']
    ]

# Get dataset
train_data, test_data, vocab_size = input_data(
    display_data=False, 
    train_frac=0.75, 
    batch=200, 
    repeat=5, 
    shuffle=200)

# Best accuracy
best_accuracy = 0.0


@use_named_args(dimensions=dimensions)
def optimize_model_fun(embedding_units, lstm_units, dense_units, dropout_rate):
    """
    Find optimal model in search space
    """
    print('----------------------------------------------------------------------')
    global best_accuracy

    # Create, train, and evaluate model
    model = create_model(vocab_size, embedding_units, lstm_units, dense_units, dropout_rate)
    model.fit(train_data, epochs=5, callbacks=[ReduceLROnPlateau(monitor='loss')])
    loss, accuracy = model.evaluate(test_data)
    print(f'Accuracy: {accuracy:0.2%}')

    # Delete model and clear session
    # this is really important!
    del model
    K.clear_session()

    # Save params if accuracy improved
    if accuracy > best_accuracy:
        print('Saving new params...')
        with open(OPTIMAL_SAVE_LOCATION, 'w') as f:
            json.dump({
                'embedding_units': embedding_units,
                'lstm_units': lstm_units,
                'dense_units': dense_units
            }, f, indent=4)
        best_accuracy = accuracy

    # Return accuracy (negative because we're minimizing)
    return -accuracy


# Run optimization
results = gp_minimize(
    func=optimize_model_fun,
    dimensions=dimensions,
    acq_func='EI',
    n_calls=11,
    x0=current_optimal_parameters)