import pymc3 as pm
import arviz as az
import numpy as np

val_T = [0.2, 0.5]
val_Y = [0, 5, 10]

p_lambda = 10 #a priori pt n

with pm.Model() as model:

    n = pm.Poisson('n', mu=p_lambda)

    for Y in val_Y:
        for theta in val_T:
            Y_observed = pm.Binomial('Y_observed', n=n, p=theta, observed=Y) #y cu distributie binomial(n,theta)
            trace = pm.sample(5000, tune=1000, random_seed=42)
            az.plot_posterior(trace, var_names=['n'], credible_interval=0.95, hdi_prob=0.95, round_to=2, point_estimate='mode',title=f'Y={Y}, θ={theta}')

az.show()
