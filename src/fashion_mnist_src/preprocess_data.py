import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import nnfs

from src.fashion_mnist_src.network import Model, Layer_Dense, Activation_ReLU, Activation_Softmax, \
    Loss_CategoricalCrossentropy, Optimizer_Adam, Accuracy_Categorical

nnfs.init()


def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test


# Create dataset
X, y, X_test, y_test = create_data_mnist('../fashion_mnist_images')
# Scale features
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
# Reshape to vectors
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Shuffle data
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 10))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=5e-5),
    accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test),
            epochs=5, batch_size=128, print_every=100)
