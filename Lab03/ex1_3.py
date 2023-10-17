'''
p(c)  - 0.05%
p(i)  - 1%
p(i|c)- 3%

alarma incendiu
p(a)  - 0.01%
p(a|c)- 2%
nu e accident
p(a|i)  - 95%
p(a|i&c)- 98%

=> p(a)= p(a|c)p(c)+p(a|i)p(i)
'''
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

model = BayesianNetwork([('C', 'I'), ('C', 'A'), ('I', 'A')])

cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.01], [0.97, 0.03]],
                          evidence=['C'], evidence_card=[2])
cpd_a = TabularCPD(variable='A', variable_card=2, 
                        values=[[0.9998, 0.02, 0.95, 0.98],
                                [0.0002, 0.98, 0.05, 0.02]],
                        evidence=['C', 'I'], evidence_card=[2, 2])

cpd_i_c = TabularCPD(variable='I', variable_card=2, values=[[0.97, 0.03], [0.9995, 0.0005]],
                     evidence=['C'], evidence_card=[2])
cpd_a_c = TabularCPD(variable='A', variable_card=2, values=[[0.98, 0.02], [0.9995, 0.0005]],
                     evidence=['C'], evidence_card=[2])
cpd_a_i = TabularCPD(variable='A', variable_card=2, values=[[0.95, 0.05], [0.99, 0.01]],
                     evidence=['I'], evidence_card=[2])

model.add_cpds(cpd_c, cpd_i_c, cpd_a_c, cpd_a_i)

assert model.check_model()

#2 cutremur dupa alarma
infer = VariableElimination(model)
result = infer.query(variables=['C'], evidence={'A': 1})
prob1 = result.values[1]

#3 incendiu fara alarma
result_i = infer.query(variables=['I'])
prob2 = result_i.values[1] * (1 - cpd_a_i.values[1, 1])

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()