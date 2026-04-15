#!/usr/bin/env python
# coding: utf-8

# # Bernstein–Vazirani Algorithm
# 
# This notebook runs the Bernstein–Vazirani algorithm on a local simulator and a QPU on Amazon Braket. The Bernstein-Vazirani algorithm can find a hidden binary string with certainty after only one call to the quantum circuit. The hidden string is releaved by querying the oracle that is a superposition of all possible binary strings.
# 
# The Bernstein–Vazirani problem is to find a hidden binary string given a black-box function that is promised to be the dot product of the input bitstring with the hidden string. It is an extension of the Deutsch-Jozsa algorithm where the function is restricted to being constant or balanced.
# 
# Typically, we would need to query the classical function $n$ times where $n$ is the length of the string. However, if the function is encoded into a quantum circuit, we need to only query the function once to reveal the hidden string. See Ref. [1] for full details.

# # References 
# [1] [Ethan Bernstein and Umesh Vazirani (1997) "Quantum Complexity Theory" SIAM Journal on Computing, Vol. 26, No. 5: 1411-1473, doi:10.1137/S0097539796300921.](https://epubs.siam.org/doi/10.1137/S0097539796300921)
# 

# # Run on a local simulator

# In[11]:


from notebook_plotting import plot_bitstrings

get_ipython().run_line_magic('matplotlib', 'inline')

from braket.devices import LocalSimulator
from braket.tracking import Tracker

from braket.experimental.algorithms.bernstein_vazirani import (
    bernstein_vazirani_circuit,
    get_bernstein_vazirani_results,
    run_bernstein_vazirani,
)

tracker = Tracker().start()  # to track Braket costs


# In[12]:


bv_circuit = bernstein_vazirani_circuit("1010")
print(bv_circuit)


# In[13]:


local_simulator = LocalSimulator()
task = run_bernstein_vazirani(bv_circuit, local_simulator, shots=1_000)
print(task)


# In[14]:


bv_results = get_bernstein_vazirani_results(task)


# In[15]:


plot_bitstrings(bv_results, title="BV counts")


# # Run on a noisy simulator
# 
# Let's try a noisy simulator

# In[16]:


from braket.circuits.noises import BitFlip, Depolarizing

noisy_bv_circuit = (
    bernstein_vazirani_circuit("001")
    .apply_gate_noise(Depolarizing(0.01))
    .apply_readout_noise(BitFlip(0.1))
)
print(noisy_bv_circuit)


# In[17]:


local_simulator = LocalSimulator("braket_dm")

task = run_bernstein_vazirani(noisy_bv_circuit, local_simulator, shots=1_000)

noisy_bv_results = get_bernstein_vazirani_results(task)


# In[18]:


plot_bitstrings(noisy_bv_results, title="Noisy BV counts")


# In[19]:


print("Task Summary")
print(f"{tracker.quantum_tasks_statistics()} \n")
print(
    f"Estimated cost to run this example: {tracker.qpu_tasks_cost() + tracker.simulator_tasks_cost():.2f} USD"
)


# Note: Charges shown are estimates based on your Amazon Braket simulator and quantum processing unit (QPU) task usage. Estimated charges shown may differ from your actual charges. Estimated charges do not factor in any discounts or credits, and you may experience additional charges based on your use of other services such as Amazon Elastic Compute Cloud (Amazon EC2).
