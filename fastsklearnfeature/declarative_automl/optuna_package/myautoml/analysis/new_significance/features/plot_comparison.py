import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

x = np.random.random(1000)
y = np.random.random(1000)

mu, sigma = 0.5, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

plt.scatter(x,y, color='blue', label='Random Sampling')
plt.scatter(x,s, color='red', label='Uncertainty Sampling')
plt.axhline(y=0.5, color='purple', label='Decision Boundary')
plt.legend(loc='lower right')
plt.show()