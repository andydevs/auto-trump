import trainer.data
from time import time

# Run preprocessor routine
print('Sampling data...')
start = time()
trainer.data.preprocess_data()
delta = time() - start
print('Total time elapsed:', delta, 's')

# Run input as test
print('Sample input...')
start = time()
trainer.data.input_data(True, 0.7, 200, 5, 200)