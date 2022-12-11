import numpy as np

# 2 -> How many tosses of the coin we do
# 0.5 -> What's the probability of tossing 1
# 10 -> Number of those coins to toss
print(np.random.binomial(2, 0.5, size=10))
