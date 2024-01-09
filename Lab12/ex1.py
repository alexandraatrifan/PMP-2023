import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

data = np.random.choice([0, 1], size=10, p=[0.3, 0.7])

grid = np.linspace(0, 1, 1000)

prior_1 = (grid <= 0.5).astype(int)

prior_2 = np.exp(-((grid - 0.5) / 0.1)**2)

posterior_1 = prior_1 * binom.pmf(np.sum(data), len(data), grid)
posterior_1 /= np.sum(posterior_1)

posterior_2 = prior_2 * binom.pmf(np.sum(data), len(data), grid)
posterior_2 /= np.sum(posterior_2)

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(grid, prior_1, label='Prior 1: Uniform(0, 0.5)')
plt.title('Prior 1')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(grid, prior_2, label='Prior 2: Bell-shaped around 0.5')
plt.title('Prior 2')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(grid, posterior_1, label='Posterior 1')
plt.title('Posterior 1')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(grid, posterior_2, label='Posterior 2')
plt.title('Posterior 2')
plt.legend()

plt.tight_layout()
plt.show()
