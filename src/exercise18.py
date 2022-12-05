import numpy as np

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.]]) # (1,3)

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
#                     x0,  x1,  x2, x3
weights = np.array([[0.2, 0.8, -0.5, 1], # (3, 4)
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T # (4, 3)

# sum weights of given input
# and multiply by the passed in gradient for this neuron
# dx0 = sum(weights[0] * dvalues[0]) # [0.2, 0.5, -0.26] * [1, 1, 1]
# dx1 = sum(weights[1] * dvalues[0])
# dx2 = sum(weights[2] * dvalues[0])
# dx3 = sum(weights[3] * dvalues[0])

# dinputs = np.array([dx0, dx1, dx2, dx3])
dinputs = np.dot(dvalues[0], weights.T)
print(dinputs)
