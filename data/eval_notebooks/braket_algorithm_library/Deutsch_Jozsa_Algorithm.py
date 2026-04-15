#!/usr/bin/env python
# coding: utf-8

# # Deutsch-Jozsa Algorithm
# 
# 
# In this notebook, we introduce one of the first quantum algorithm’s developed by pioneers David Deutsch and Richard Jozsa. This algorithm showcases an efficient quantum solution to a problem that cannot be solved classically but instead can be solved using a quantum device [1].
# 
# ## References 
# [1] [Michael A. Nielsen & Isaac L. Chuang Quantum Computation and Quantim Information textbook](http://mmrc.amss.cas.cn/tlb/201702/W020170224608149940643.pdf)
# 
# 
# ## Background
# Let $U_f$ be an oracle (black box) that computes a Boolean function which only takes binary inputs (0’s or 1’s). These functions can be represented as $f: \{0, 1\}^n →  \{0, 1\}$. This oracle evaluates two types of functions, constant or balanced. 
# 
# A constant function takes any input and returns only 0’s or only 1’s and a balanced function takes any input and returns exactly half 0’s and half 1’s. 
# 
# **Constant:** 011010 → $U_f$ → 000000  
# **Balanced:** 000011 → $U_f$ → 111000
# 
# The goal is to determine what type of function is $U_f$ based on only the outputs. If the input is as large as $2^n$ then the amount of queries a classical computer will have to make is $2^(n - 1)+1$. We can see for a large enough $n$ this problem scales exponentially and becomes inefficient to solve classically. However, leveraging a quantum algorithm we only need to query the oracle once to determine the type of function for $U_f$. This is possible because the state of its output might be in a coherent superposition of states corresponding to different answers, each which solves the problem.
# 

# In[1]:


from notebook_plotting import plot_bitstrings

get_ipython().run_line_magic('matplotlib', 'inline')


from braket.experimental.algorithms.deutsch_jozsa import (
    balanced_oracle,
    constant_oracle,
    deutsch_jozsa_circuit,
    get_deutsch_jozsa_results,
)

from braket.tracking import Tracker

tracker = Tracker().start()  # to track Braket costs


# 
# The initialization function prepares a superposition of all possible input values and the second register is in a superposition of 0 and 1. 

# In[2]:


n_qubits = 3


# The constant oracle circuit is shown below:

# In[3]:


const_oracle = constant_oracle(n_qubits)
print(const_oracle)


# The balanced oracle circuit is shown below:

# In[4]:


bal_oracle = balanced_oracle(n_qubits)
print(bal_oracle)


# The final circuit for the solved Deutsch-Jozsa problem is below:

# In[5]:


dj_circuit = deutsch_jozsa_circuit(bal_oracle)
print(dj_circuit)


# In[6]:


dj_circuit = deutsch_jozsa_circuit(const_oracle)
print(dj_circuit)


# The oracle is the Boolean function that is applied to the $n$-qubits in the query register. 

# ## Run on a local simulator
# 
# If the output is "000", then the algorithm predicts a constant oracle. If the output is "111", it predicts balanced.

# In[7]:


from braket.devices import LocalSimulator

device = LocalSimulator()


# In[8]:


task = device.run(dj_circuit, shots=10_000)


# We can get the results with 

# In[9]:


dj_probabilities = get_deutsch_jozsa_results(task)
print(dj_probabilities)


# In[10]:


plot_bitstrings(dj_probabilities)


# We see that the probability of "000" is one, so the results correctly indicate that the oracle was constant.

# In[11]:


print("Task Summary")
print(f"{tracker.quantum_tasks_statistics()} \n")
print(
    f"Estimated cost to run this example: {tracker.qpu_tasks_cost() + tracker.simulator_tasks_cost():.2f} USD"
)


# Note: Charges shown are estimates based on your Amazon Braket simulator and quantum processing unit (QPU) task usage. Estimated charges shown may differ from your actual charges. Estimated charges do not factor in any discounts or credits, and you may experience additional charges based on your use of other services such as Amazon Elastic Compute Cloud (Amazon EC2).
