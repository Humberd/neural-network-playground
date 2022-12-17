import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.fashion_mnist_src.network import Model

image_data = cv2.imread('../fashion_mnist_images/real/tshirt.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(image_data, cmap='gray')
plt.show()
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data
plt.imshow(image_data, cmap='gray')
plt.show()

image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
# Load the model
model = Model.load('fashion_mnist.model')
# Predict on the image
confidences = model.predict(image_data)
# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)
