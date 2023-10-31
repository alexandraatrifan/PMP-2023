import pymc3 as pm
import numpy as np
import pandas as pd

data = pd.read_csv('trafic.csv', header=0, names=['minute', 'trafic'])

intervale = [(7, 8), (8, 9), (16, 17), (19, 20), (20, 21)]

if __name__ == '__main__':
    with pm.Model() as model:

        lamb = pm.Exponential('lamb', lam=1)
        trafic = pm.Poisson('trafic', mu=lamb, observed=data['trafic'])
        trace = pm.sample(10000, tune=1000, chains=4)

medii_lamb = []
for interval in intervale:
    intervale_data = data[(data['minute'] >= interval[0]*60) & (data['minute'] < interval[1]*60)]
    medie_trafic = intervale_data['trafic'].mean()
    medii_lamb.append(medie_trafic)

for i, interval in enumerate(intervale):
    print(f'Intervalul {interval}: Estimare lamb = {medii_lamb[i]:.2f}')

pm.summary(trace, var_names=['lamb'])
