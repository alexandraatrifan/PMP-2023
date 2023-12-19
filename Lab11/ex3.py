import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneOut

means = [5, 0, -5]
std_devs = [2, 2, 2]
weights = [0.4, 0.3, 0.3]

num_samples = 500
data = np.concatenate([np.random.normal(mean, std_dev, int(weight * num_samples))
    for mean, std_dev, weight in zip(means, std_devs, weights)])
data = data.reshape(-1, 1)

models = {}
for n_components in [2, 3, 4]:
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(data)
    models[n_components] = model

log_likelihoods = {}
for n_components, model in models.items():
    log_likelihoods[n_components] = model.score_samples(data)

waic_results = {}
for n_components, ll in log_likelihoods.items():
    n_data_points = len(data)
    waic_i = -2 * (np.sum(np.log(np.mean(np.exp(ll), axis=0))) - np.mean(ll))
    waic_results[n_components] = waic_i + 2 * np.sum(np.var(ll))

loo_results = {}
for n_components, ll in log_likelihoods.items():
    loo = LeaveOneOut()
    loo_i = np.zeros(n_data_points)

    for train_index, test_index in loo.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_ll = model.score_samples(train_data)
        loo_i[test_index] = model.score_samples(test_data) - np.mean(train_ll)

    loo_results[n_components] = np.sum(loo_i)


for n_components, waic_value in waic_results.items():
    print(f"Model WAIC cu {n_components} componente: {waic_value}")

print("\n")
for n_components, loo_value in loo_results.items():
    print(f"Model LOO cu {n_components} componente: {loo_value}")
