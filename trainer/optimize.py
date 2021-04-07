"""
Find optimal model dimensions using scikit optimize
"""
from .model import create_model
from .data import input_data
from skopt import gp_minimize
from skopt.space import Integer
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
dimensions = [ dim_embedding_units, dim_lstm_units, dim_dense_units ]
default_parameters = [ 250, 500, 750 ]

# Get dataset
train_data, test_data, vocab_size = input_data(False, 0.75, 200, 5, 200)


@use_named_args(dimensions=dimensions)
def optimize_model_fun(embedding_units, lstm_units, dense_units):
    """
    Find optimal model in search space
    """
    # Print parameters
    print('----------------------------------------------------------------------')
    print('Embedding units:', embedding_units)
    print('LSTM units:', lstm_units)
    print('Dense units:', dense_units)

    # Create, train, and evaluate model
    model = create_model(vocab_size, embedding_units, lstm_units, dense_units)
    model.fit(train_data, epochs=10, callbacks=[ReduceLROnPlateau(monitor='loss')])
    loss, accuracy = model.evaluate(test_data)
    print(f'Accuracy: {accuracy:0.2%}')

    # Delete model and clear session
    # this is really important!
    del model
    K.clear_session()

    # Return accuracy (negative because we're minimizing)
    return -accuracy


# Run optimization
results = gp_minimize(
    func=optimize_model_fun,
    dimensions=dimensions,
    acq_func='EI',
    n_calls=40,
    x0=default_parameters)

# Save optimal parameters
with open(OPTIMAL_SAVE_LOCATION, 'w') as f:
    json.dump({
        dimension.name:str(results.x[index])
        for index, dimension in enumerate(results.space)
    }, f)