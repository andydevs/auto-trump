import imp
import trainer.data
from time import time
start = time()
trainer.data.input_data(True, 0.7, 200, 3, True)
delta = time() - start
print('Total time elapsed:', delta, 's')