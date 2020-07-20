# The starting script of the Agent-Based Model of Interbank
# Payment System. Version on 19 July 2020.


import collections
import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import model


# Global parameters.
type_v = [100000, 10000, 1000]   # Fixed bank types.
n = [5, 20, 75]   # Number of banks of each type.
day_length = 570
max_length = 10000
i_rate = 1.0   # Interest rate on the market.
d_rate = 0.0001   # Deferral rate.
i_night = 0.0015   # Overnight interest rate.
d_night = 0.0015   # Overnight deferral rate
p_table = np.asarray([[0.95, 0.05, 0.0 ],
		      [0.5 , 0.45, 0.05],
		      [0.0 , 0.5 , 0.5 ]])   # Bank-payer in rows.

# Main run.
banks, trace = run_model()

# Calculates average for bank-types.
def_rate_avr = np.zeros([3,day_length])
def_rate_avr[0,:] = np.sum(trace.def_rate[:5,:],0)/5
def_rate_avr[1,:] = np.sum(trace.def_rate[5:25,:],0)/20
def_rate_avr[2,:] = np.sum(trace.def_rate[25:,:],0)/75

# Plotting of averages. First 50 time periods
plt.figure(figsize=(12, 6))
plt.plot(def_rate_avr[0,:50], label='Large banks')
plt.plot(def_rate_avr[1,:50], label='Medium banks')
plt.plot(def_rate_avr[2,:50], label='Small banks')
plt.legend(loc='lower right')
plt.show()

# Preparing data for network.
edge_list = []
for i in range(sum(n)):
    for j in range(sum(n)):
        if trace.pay_trace[i,j,0] != 0:
            if trace.pay_trace[i,j,2]/trace.pay_trace[i,j,0] == 1:
                edge_list.append((i,j))
node_list = range(sum(n))

# Plotting the network.
plt.figure(figsize=[15,10])
G = nx.Graph()
G.add_nodes_from(node_list)
G.add_edges_from(edge_list)
nx.draw(G, with_labels=True)
plt.show()
