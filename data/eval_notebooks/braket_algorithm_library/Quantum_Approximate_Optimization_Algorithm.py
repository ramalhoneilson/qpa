#!/usr/bin/env python
# coding: utf-8

# # Quantum Approximate Optimization Algorithm (QAOA)

# In this notebook, we show how to (approximately) solve binary combinatorial optimization problems, using the __Quantum Approximate Optimization Algorithm (QAOA)__.
# 
# The QAOA is a variational quantum algorithm that uses alternating layers of parameterized quantum gates to solve an optimization problem [1]. The parameters of each gate are tuned to minimize a cost function, similar to how machine learning parameters are tuned in stochastic gradient descent. 
# 
# ## References 
# [1] Edward Farhi, Jeffrey Goldstone, and Sam Gutmann, "A Quantum Approximate Optimization Algorithm Applied to a Bounded Occurrence Constraint Problem," (2014), [arXiv:1412.6062](https://arxiv.org/abs/1411.4028)

# In[1]:


import numpy as np
from braket.devices import LocalSimulator
from scipy.optimize import minimize

from braket.experimental.algorithms.quantum_approximate_optimization import cost_function, qaoa
from braket.tracking import Tracker

tracker = Tracker().start()  # to track Braket costs


# In[2]:


n_qubits = 2
n_layers = 1

coupling_matrix = np.random.rand(n_qubits, n_qubits)
np.fill_diagonal(coupling_matrix, 0)

print(coupling_matrix)


# In[3]:


circ = qaoa(n_qubits, n_layers, coupling_matrix)
print(circ)


# In[4]:


idx = coupling_matrix.nonzero()
coeffs = [coupling_matrix[qubit_pair] for qubit_pair in zip(idx[0], idx[1])]
print(coeffs)


# ## Run on a local simulator
# 
# Now we run the QAOA on a local simulator by the Nelder-Mead method from scipy.optimize.

# In[5]:


device = LocalSimulator()

init_values = np.random.rand(2 * n_layers)

# set bounds for search space
bounds = [(0, 2 * np.pi) for _ in range(2 * n_layers)]

losses = []


# In[6]:


losses = []
result = minimize(
    cost_function,
    init_values,
    args=(device, circ, coeffs, losses, 0),  # shots=0
    options={"disp": True, "maxfev": 150},
    method="Nelder-Mead",
    # bounds=bounds, # optional, some optimizers can use bounds
)


# In[7]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(losses, "-o")
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.title("QAOA convergence of cost function")


# In[8]:


print("Task Summary")
print(f"{tracker.quantum_tasks_statistics()} \n")
print(
    f"Estimated cost to run this example: {tracker.qpu_tasks_cost() + tracker.simulator_tasks_cost():.2f} USD"
)


# Note: Charges shown are estimates based on your Amazon Braket simulator and quantum processing unit (QPU) task usage. Estimated charges shown may differ from your actual charges. Estimated charges do not factor in any discounts or credits, and you may experience additional charges based on your use of other services such as Amazon Elastic Compute Cloud (Amazon EC2).
