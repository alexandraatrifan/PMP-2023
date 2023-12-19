import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

means = [5, 0, -5]
std_devs = [2, 2, 2]
weights = [0.4, 0.3, 0.3]

num_samples = 500
data = np.concatenate([np.random.normal(mean, std_dev, int(weight * num_samples))
    for mean, std_dev, weight in zip(means, std_devs, weights)])
data = data.reshape(-1, 1)

for n_components in [2, 3, 4]:
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(data)

    plt.hist(data, bins=30, density=True, alpha=0.5, color='b')
    x = np.linspace(min(data), max(data), 1000)
    y = np.exp(model.score_samples(x.reshape(-1, 1)))
    plt.plot(x, y, color='red', lw=2, label=f'{n_components} componente')
    plt.title(f'Model de mixtura Gausiana cu {n_components} componente')
    plt.xlabel('valoare')
    plt.ylabel('densitate')
    plt.legend()
    plt.show()
