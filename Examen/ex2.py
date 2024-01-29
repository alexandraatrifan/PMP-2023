import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def estimate_P(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = x > (y ** 2)
    prob = inside.sum() * 4 / N

    error = abs((prob - np.prob) / prob) * 100

    return error

runs = 10000 # numarul de iteratii
N = stats.geom.rvs(p = 0.5, size=1000)

#se ruleaza estimate_P(N) de runs ori
estimates = np.arange(0)
for _ in range(runs):
    estimates = np.concatenate((estimates,(estimate_P(N),)))

#b)
#listele pentru a stoca media si deviatia standard
mean_errors = [0,0,0]
std_errors = [0,0,0]
k = 30

for i in range(k):
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