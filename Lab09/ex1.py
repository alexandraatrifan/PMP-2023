import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('Admission.csv')

    data['GRE'] = pd.to_numeric(data['GRE'], errors='coerce')
    data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
    gre_scores = data['GRE']
    gpa_scores = data['GPA']

    gre_mean, gre_std = gre_scores.mean(), gre_scores.std()
    gpa_mean, gpa_std = gpa_scores.mean(), gpa_scores.std()

    gre_scores = (gre_scores - gre_mean) / gre_std
    gpa_scores = (gpa_scores - gpa_mean) / gpa_std

    with pm.Model() as logistic_model:
        beta_0 = pm.Normal('beta_0', mu=0, sd=1)
        beta_1 = pm.Normal('beta_1', mu=0, sd=1)
        beta_2 = pm.Normal('beta_2', mu=0, sd=1)

        admit_prob = pm.invlogit(beta_0 + beta_1 * gre_scores + beta_2 * gpa_scores)
        admission = pm.Bernoulli('admission', p=admit_prob, observed=data['Admission'])

    with logistic_model:
        trace = pm.sample(2000, tune=1000, chains=2)

    pm.summary(trace)


    pm.plot_posterior(trace, var_names=['beta_1', 'beta_2'], color='LightSeaGreen', ref_val=0)

#pct2
    plt.figure()
    plt.scatter(gre_scores, gpa_scores, c=data['Admission'], cmap='viridis', alpha=0.7)
    plt.xlabel('GRE Scores')
    plt.ylabel('GPA Scores')

    plt.show()

#pct3
    new_data_1 = {'GRE': (550 - gre_mean) / gre_std, 'GPA': (3.5 - gpa_mean) / gpa_std}
    with logistic_model:
        post_pred_1 = pm.sample_posterior_predictive(trace, samples=2000, var_names=['admission'], new_data=new_data_1)

    hdi_1 = pm.hpd(post_pred_1['admission'], hdi_prob=0.9)

#pct4
    new_data_2 = {'GRE': (500 - gre_mean) / gre_std, 'GPA': (3.2 - gpa_mean) / gpa_std}
    with logistic_model:
        post_pred_2 = pm.sample_posterior_predictive(trace, samples=2000, var_names=['admission'], new_data=new_data_2)

    hdi_2 = pm.hpd(post_pred_2['admission'], hdi_prob=0.9)

    print("intervalul HDI cu prob de admitere (GRE=550, GPA=3.5):", hdi_1)
    print("intervalul HDI cu prob de admitere (GRE=500, GPA=3.2):", hdi_2)
