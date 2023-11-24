from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt

# p(J0)=1/2 P(J1)=1/2
# P(S|J0) = 1/2
# p(S|j1) = 2/3

simulare = 10000

# simulez jocul de 10000 ori
while simulare != 0:
    # structura retelei bayesiene
    model = BayesianNetwork([('jucator0', 'stema'), ('jucator1', 'stema')])

    # definirea distributiei conditionate pt fiecare variabila
    jucator0 = TabularCPD(variable='jucator0', variable_card=2,
                        values=[[0.5], [0.5]])

    jucator1 = TabularCPD(variable='jucator1', variable_card=2,
                        values=[[0.5], [0.5]])
    stema = TabularCPD(variable='stema', variable_card=2,
                        values=[[0.5, 0.666], [0.5, 0.334]],
                        evidence=['jucator0', 'jucator1'], evidence_card=[2, 2])
    model.add_cpds(jucator0, jucator1, stema)
    assert model.check_model() #verific daca e valid modelul
    infer = VariableElimination(model)

    stema_moneda0 = stats.binom.rvs(1,0.5, size=10) #nemasluita
    stema_moneda1 = stats.binom.rvs(1,0.666, size=10) #masluita

    # calculez P(S|J0) si P(S|J1)
    # cand j0 e primul, j1 al doilea    prob_stema_j0_I = infer.query(variables=['stema'], evidence={'jucator0': 1})
    prob_stema_j0_I = infer.query(variables=['stema'], evidence={'jucator0s': 1})
    if stema_moneda0 == 0:
        prob_stema_j1_II = infer.query(variables=['stema'], evidence={'jucator1': 0})
    else:
        prob_stema_j1_II = infer.query(variables=['stema'], evidence={'jucator1': 0})*2

    # cand j1 e primul, j0 al doilea
    prob_stema_j1_I = infer.query(variables=['stema'], evidence={'jucator1': 1})
    if stema_moneda1 == 0:
        prob_stema_j0_II = infer.query(variables=['stema'], evidence={'jucator0': 0})
    else:
        prob_stema_j0_II = infer.query(variables=['stema'], evidence={'jucator0': 0})*2

    prob_stema_j0 = prob_stema_j0_I * prob_stema_j0_II
    prob_stema_j1 = prob_stema_j0_I * prob_stema_j0_II

    print(f"J0: {prob_stema_j0} si J1: {prob_stema_j1}")

    # compar cele doua probabilitati pentru a stabili cine are sanse mai mari de castig
    if prob_stema_j0 < prob_stema_j1:
        print("J1 are sanse mai mari de castig")
    else:
        print("J0 are sanse mai mari de castig")
    simulare -= 1

# afisare
pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()