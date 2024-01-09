import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return error

N_values = [100, 1000, 10000]

errors = [estimate_pi(N) for N in N_values]

mean_error = np.mean(errors)
std_dev_error = np.std(errors)

plt.errorbar(N_values, errors, yerr=std_dev_error, fmt='o-', capsize=5)
plt.xscale('log')
plt.xlabel('Nr de puncte(N)')
plt.ylabel('Eroare(%)')
plt.title('Estimare π')
plt.show()
