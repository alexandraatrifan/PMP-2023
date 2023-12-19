import numpy as np
import matplotlib.pyplot as plt

means = [5, 0, -5]
std_devs = [2, 2, 2]
weights = [0.4, 0.3, 0.3]

num_samples = 500
data = np.concatenate([np.random.normal(mean, std_dev, int(weight * num_samples))
    for mean, std_dev, weight in zip(means, std_devs, weights)])

np.random.shuffle(data)

plt.hist(data, bins=30, density=True, alpha=0.5, color='b')
plt.title('Amestec de 3 distributii gausiene')
plt.xlabel('valoare')
plt.ylabel('densitate')
plt.show()
