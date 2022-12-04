import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

layer_outputs = [4.8, 1.21, 2.385]

E = 2.71828182846

exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)

norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)

print('exponentiated values:')
print(exp_values)

print('Normalized exponentiated values:')
print(norm_values)
print('Sum of normalized values:', sum(norm_values))
