import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

#a) incarcarea datelor
supravietuire = pd.read_csv("Titanic.csv")
y = supravietuire["Survived"]
x_Pclass = supravietuire["Pclass"].values
x_Age = supravietuire["Age"].values
x_Pclass_mean = x_Pclass.mean()
x_Age_mean = x_Age.mean()
x_Pclass_std = x_Pclass.std()
x_Age_std = x_Age.std()

#standardizarea datelor
x_Age = (x_Age-x_Age_mean)/x_Age_std
x_Pclass = (x_Pclass-x_Pclass_mean)/x_Pclass_std
X = np.column_stack((x_Pclass,x_Age))

#b)
with pm.Model() as surv_model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1, shape = 2)
    X_shared = pm.MutableData('x_shared',X)
    mu = pm.Deterministic('μ',alpha + pm.math.dot(X_shared, beta)) # modelul logistic
    theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
    y_pred = pm.Bernoulli("y_pred", p=theta, observed=y)
    idata = pm.sample(2000, return_inferencedata = True)

#c) Age a influentat cel mai mult daca a supravietuit sau nu
idx = np.sort(x_Age)
plt.scatter(x_Age, x_Pclass, c=[f"C{x}" for x in y])
plt.xlabel("Age")
plt.xlabel("Passenger Class")

#d) Age = 30; Pclass = 2; HDI = 90%
pass_1 = [(2-x_Pclass_mean)/x_Pclass_std, (30-x_Age_mean)/x_Age_std]
pm.set_data({"x_shared":[pass_1]}, model=surv_model) # actualizare date model
ppc = pm.sample_posterior_predictive(idata, model=surv_model, var_names=["theta"]) # rezultatele simularii

#vizualizarea intervalului de incredere HDI
y_ppc = ppc.posterior_predictive['theta'].stack(sample=("chain", "draw")).values
az.plot_posterior(y_ppc,hdi_prob=0.9)
