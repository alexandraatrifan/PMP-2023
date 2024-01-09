import numpy as np
from scipy.stats import beta as beta_distribution
import matplotlib.pyplot as plt


def prior_beta(alpha, beta, theta):
    if 0 <= theta <= 1:
        return beta_distribution.pdf(theta, alpha, beta)
    else:
        return 0


def metropolis(current_alpha, current_beta, n, k, proposal_std, iterations):
    samples_alpha = [current_alpha]
    samples_beta = [current_beta]

    for _ in range(iterations):
        prop_alpha = abs(np.random.normal(current_alpha, proposal_std))
        prop_beta = abs(np.random.normal(current_beta, proposal_std))

        current_theta = np.random.beta(current_alpha, current_beta)
        prop_theta = np.random.beta(prop_alpha, prop_beta)

        acceptance_ratio = (
            prior_beta(prop_alpha, prop_beta, current_theta) /
            prior_beta(current_alpha, current_beta, current_theta) *
            prior_beta(current_alpha, current_beta, prop_theta) /
            prior_beta(prop_alpha, prop_beta, prop_theta)
        )

        if np.random.uniform(0, 1) < acceptance_ratio:
            current_alpha, current_beta = prop_alpha, prop_beta

        samples_alpha.append(current_alpha)
        samples_beta.append(current_beta)

    return np.array(samples_alpha), np.array(samples_beta)


n_observed = 50
k_observed = 25
proposal_std = 0.1
iterations = 10000

alpha_prior = 2
beta_prior = 5

alpha_samples, beta_samples = metropolis(alpha_prior, beta_prior, n_observed, k_observed, proposal_std, iterations)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(alpha_samples)
plt.title('Evolutie alpha')

plt.subplot(1, 2, 2)
plt.plot(beta_samples)
plt.title('Evolutie beta')

plt.show()