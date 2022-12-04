import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 2 * x ** 2


x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y)

for i in range(5):
    p2_delta = 0.0001
    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    approximate_derivative = (y2 - y1) / (x2 - x1)
    # y = mx+b
    b = y2 - approximate_derivative * x2
    print(b)


    def tangent_line(x):
        # y = mx+b
        return approximate_derivative * x + b


    to_plot = [x1 - 0.9, x1, x1 + 0.9]
    plt.scatter(x1, y1)
    plt.plot(to_plot, [tangent_line(i) for i in to_plot])

    print('Approximate derivative for f(x)', f'where x = {x1} is {approximate_derivative}')

plt.show()
