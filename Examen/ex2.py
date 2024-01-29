import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymc as pm

def estimate_P(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = x > (y ** 2)
    prob = inside.sum() * 4 / N

    error = abs((prob - N) / prob) * 100

    return error

runs = 10000 # numarul de iteratii
theta = pm.Geometric("theta")
N_values = stats.geom.rvs(theta)

#se ruleaza estimate_P(N) de runs ori
for i,N in enumerate(N_values):
    estimates = np.arange(0)
    for _ in range(runs):
        estimates = np.concatenate((estimates,(estimate_P(N),)))
        print(f"N: {N}; estimates: {estimates}\n")

#b)
#listele pentru a stoca media si deviatia standard
mean_errors = [0,0,0]
std_errors = [0,0,0]
k = 30

for i in range(k):
    N = np.random.uniform(N_values)
    estimates = np.arange(0)
    for _ in range(runs):
        estimates = np.concatenate((estimates,(estimate_P(N),)))
    mean_errors[i] = np.mean(estimates)
    std_errors[i] = np.std(estimates)

plt.errorbar(N, mean_errors, yerr=std_errors, fmt='o-')
plt.xlabel('Numarul de puncte')
plt.ylabel('Eroare relativa')
plt.title('Estimarea erorii relative in functie de numarul de puncte')
plt.show()