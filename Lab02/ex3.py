import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az


stema = 0.3 #probabilitate stema 
cap = 0.7 #probabilitatile cap
nr = 10 #aruncari
exp = 100 #experimente

#fct pentru aruncare
def aruncare(stema):
    if np.random.random() <= stema:
        return 's' 
    else:
        return 'b'

#fct pentru experiment
def experiment():
    rezultat = ""
    for i in range(1,nr):
        rezultat = rezultat + aruncare(stema) + aruncare(cap)
    return rezultat

rezultate = {'ss': 0, 'sb': 0, 'bs': 0, 'bb': 0}

#fac experimentele
for i in range(1,exp):
    rezultat = experiment()
    rezultate[rezultat] = rezultate[rezultat] + 1

az.plot_posterior(rezultate)
plt.show()