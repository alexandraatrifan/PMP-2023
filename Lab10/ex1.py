import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

#modelul polinomial din curs sub forma de functie
def data(order_d, size_d):
    dummy_data = np.loadtxt('dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]

    x_1p = np.vstack([x_1**i for i in range(1, order_d+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    return x_1s[0][:size_d], y_1s[:size_d]

# Exercițiul 1
order = 5
size = 100
x, y = data(order, size)

with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order + 1)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y)
    trace_p = pm.sample(2000, tune=1000)

x_new = np.linspace(0, 1, 100)
y_new = trace_p['α'] + np.dot(trace_p['β'], x_new[:, None].T)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='date observate')
plt.plot(x_new, y_new[:, :], color='red', alpha=0.1)
plt.title(f'model polinomial de ordin 5')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

az.summary(trace_p, hdi_prob=0.95)

x_new = np.linspace(0, 1, 100)
y_new = trace_p['α'] + np.dot(trace_p['β'], x_new[:, None].T)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Date observate')
plt.plot(x_new, y_new[:, :], color='red', alpha=0.1)
plt.title(f'model polinomial de ordin 5 - sd=100')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

x_new = np.linspace(0, 1, 100)
y_new = trace_p['α'] + np.dot(trace_p['β'], x_new[:, None].T)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Date observate')
plt.plot(x_new, y_new[:, :], color='red', alpha=0.1)
plt.title(f'model polinomial de ordin 5 - sd=np.array([10, 0.1, 0.1, 0.1, 0.1])')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# Exercițiul 2
order = 5
size = 500
x, y = data(order, size)

with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order + 1)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y)
    trace_p = pm.sample(2000, tune=1000)

x_new = np.linspace(0, 1, 100)
y_new = trace_p['α'] + np.dot(trace_p['β'], x_new[:, None].T)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Date observate')
plt.plot(x_new, y_new[:, :], color='red', alpha=0.1)
plt.title(f'model polinomial de ordin 5 cu 500 de puncte')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

az.summary(trace_p, hdi_prob=0.95)


# Exercițiul 3
order = 3
x, y = data(order, size)

with pm.Model() as model_cubic:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order + 1)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y)
    trace_cubic = pm.sample(2000, tune=1000)

az.summary(trace_cubic, hdi_prob=0.95)

x_new = np.linspace(0, 1, 100)
y_new_cubic = trace_cubic['α'] + np.dot(trace_cubic['β'], x_new[:, None].T)

plt.figure(figsize=(12, 8))
plt.scatter(x, y, label='date observate', alpha=0.6)
plt.plot(x_new, y_new_cubic[:, :], color='green', alpha=0.1, label='model cubic')
plt.plot(x_new, y_new[:, :], color='red', alpha=0.1, label='model patratic')
plt.plot(x_new, trace_p['α'].mean() + np.dot(trace_p['β'].mean(axis=0), x_new[:, None].T), color='blue', alpha=0.8, label='model liniar')
plt.title(f'Comparare model liniar, patratic si cubic')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
