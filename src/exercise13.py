import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

# Probabilities of 3 samples
softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])
# Target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])

# result: [0 0 1]
predictions = np.argmax(softmax_outputs, axis = 1)

# If targets are one-hot encoded - convert them
if (len(class_targets.shape) == 2):
    class_targets = np.argmax(class_targets, axis = 1)

accuracy = np.mean(predictions == class_targets)

print(predictions)
print(predictions == class_targets)
print('acc:', accuracy)
