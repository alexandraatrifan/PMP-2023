import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('auto-mpg.csv')
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df = df.dropna(subset=['horsepower', 'mpg'])

    plt.figure(figsize=(10, 6))
    plt.scatter(df['horsepower'], df['mpg'])
    plt.title('Relatia de Dependenta')
    plt.xlabel('cai putere')
    plt.ylabel('consum')
    plt.show()

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=10)

        hp = pm.Data('hp', df['horsepower'])
        mu = alpha + beta * hp

        sigma = pm.HalfNormal('sigma', sd=1)
        mpg = pm.Normal('mpg', mu=mu, sd=sigma, observed=df['mpg'])

    with model:
        map_aprox = pm.find_MAP()
        map_alpha = map_aprox['alpha']
        map_beta = map_aprox['beta']
        print(f'Dreapta de regresie: y = {map_alpha:.2f} + {map_beta:.2f} * HP')

    x_val = np.linspace(min(df['horsepower']), max(df['horsepower']), 100)
    y_val = map_alpha + map_beta * x_val

    plt.scatter(df['horsepower'], df['mpg'])
    plt.plot(x_val, y_val, color='red')
    plt.title('Regresie')
    plt.xlabel('cai putere')
    plt.ylabel('consum')
    plt.show()

#concluzie d) modelul este bun cand regiunea e ingusta si se suprapune pe date reale