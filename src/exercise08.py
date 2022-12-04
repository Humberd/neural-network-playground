import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values)

print('exponentiated values:')
print(exp_values)

print('Normalized exponentiated values:')
print(norm_values)
print('Sum of normalized values:', sum(norm_values))
