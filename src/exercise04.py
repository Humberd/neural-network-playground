import numpy as np

# number of nodes = 4
inputs = [
    [1.0, 2.0, 3.0, 2.5], # dataset 1
    [2.0, 5.0, -1.0, 2.0], # dataset 2
    [-1.5, 2.7, 3.3, -0.8] # dataset 3
]
weights = [
    [0.2, 0.8, -0.5, 1], # weights from input layer: node1, node2, node3, node4 connected to node1 of hidden_layer1
    [0.5, -0.91, 0.26, -0.5], # weights from input layer: node1, node2, node3, node4 connected to node2 of hidden_layer1
    [-0.26, -0.27, 0.17, 0.87] # weights from input layer: node1, node2, node3, node4 connected to node3 of hidden_layer1
]
biases = [2, 3, 0.5] # biases per node
weights2 = [
    [0.1, -0.14, 0.5], # weights from hidden_layer1: node1, node2, node3 connected to node1 of hidden_layer2
    [-0.5, 0.12, -0.33], # weights from hidden_layer1: node1, node2, node3 connected to node2 of hidden_layer2
    [-0.44, 0.73, -0.13] # weights from hidden_layer1: node1, node2, node3 connected to node3 of hidden_layer2
]
biases2 = [-1, 2, -0.5]

layer_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer_outputs, np.array(weights2).T) + biases2

print(layer_outputs)
print(layer2_outputs)
