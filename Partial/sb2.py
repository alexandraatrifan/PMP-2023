import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az
from scipy import stats
import copy

medie = []
for i in range(100):
    nr_serviri = stats.expon.rvs(loc=3, size=20).mean()
    medie.append(copy.deepcopy(nr_serviri))

with pm.Model() as model:
    miu = 0
    sigma = 3
    timp = pm.Normal("timp", miu, sigma) #distributia a priori
    observation = pm.Poisson("obs", mu=timp, observed = medie)
    idata_t = pm.sample(100, return_inferencedata=True, cores=1) #obt distributia a posteriori

    with model:
        trace = pm.sample(1000, cores=1)
        az.plot_posterior(trace)
        plt.show()