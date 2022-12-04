import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
elif len(class_targets.shape) == 2:
    # it multiplies [0.7, 0.1, 0.2] * [1, 0, 0], which is (0.7 * 1) + (0.1 * 0) + (0.2 * 0), etc
    correct_confidences =  np.sum(softmax_outputs * class_targets, axis=1)

print(correct_confidences)
neg_log = -np.log(correct_confidences)
average_loss = np.mean(neg_log)
print(average_loss)

print(-np.log(1.5))
