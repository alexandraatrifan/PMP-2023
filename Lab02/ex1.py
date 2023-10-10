import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

lam1 = 4
lam2 = 6

#timpul
x1 = np.random.exponential(scale=1/lam1, size=10000)
x2 = np.random.exponential(scale=1/lam2, size=10000)

#media
m1 = stats.expon(0,1/lam1)
m2 = stats.expon(0,1/lam2)

#deviatia
d1 = np.std(x1)
d2 = np.std(x2)

x = stats.norm.rvs(d1, m1, size=10000) 
y = stats.uniform.rvs(d2, m2, size=10000) 
z = x+y 

az.plot_posterior({'x':x,'y':y,'z':z}) 
plt.show() 
