#!/usr/bin/env python
# coding: utf-8

# # Simon's Algorithm
# 
# This notebook demonstrates the execution of Simon's algorithm on Amazon Braket. 
# 
# 
# Suppose we’re given a function $f:\{0,1\}^n \rightarrow \{0,1\}^n$ that maps bit strings to bit strings along with the promise that
# $$\forall x,y \in \{0,1\}^n, \quad f(x) = f(y) \iff x=y\oplus s,$$
# for some unknown $n$-bit string $s \in \{0,1\}^n$, and where $\oplus$ means bitwise addition modulo 2.
# 
# The goal of Simon's problem is to determine if $f$ is one-to-one, or two-to-one, or equivalently to find the secret string $s$.

# # References 
# [1] Wikipedia: [Simon's Problem](https://en.wikipedia.org/wiki/Simon%27s_problem)
# 
# [2] SIAM: [Original article](https://epubs.siam.org/doi/10.1137/S0097539796298637)
# 
# [3] Amazon Braket's GitHub repository: [Detailed example notebook](https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/advanced_circuits_algorithms/Simons_Algorithm/Simons_Algorithm.ipynb)

# # Run on a local simulator

# In[1]:


import numpy.random as npr
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.tracking import Tracker

from braket.experimental.algorithms.simons import (
    get_simons_algorithm_results,
    run_simons_algorithm,
    simons_algorithm,
    simons_oracle,
)

tracker = Tracker().start()


# In[2]:


string_length = 5
string = "".join(npr.choice(["0", "1"], size=string_length))

oracle = simons_oracle(string)


# # Print the circuits

# In[3]:


print(simons_algorithm(oracle))


# In[4]:


local_simulator = LocalSimulator()

task = run_simons_algorithm(oracle, local_simulator)


# In[5]:


processed_results = get_simons_algorithm_results(task)
print("Secret string", string)


# # Run on a QPU
# 
# Here we run Simon's algorithm on the Rigetti Ankaa-2 device.

# In[6]:


ankaa = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2")


# In[7]:


# run this block to submit the circuit to QPU

task = run_simons_algorithm(oracle, ankaa)


# In[8]:


# run this block to process the results

processed_results = get_simons_algorithm_results(task)
print("Secret string", string)


# In[9]:


print("Task Summary")
print(f"{tracker.quantum_tasks_statistics()} \n")
print(
    f"Estimated cost to run this example: {tracker.qpu_tasks_cost() + tracker.simulator_tasks_cost():.2f} USD"
)


# Note: Charges shown are estimates based on your Amazon Braket simulator and quantum processing unit (QPU) task usage. Estimated charges shown may differ from your actual charges. Estimated charges do not factor in any discounts or credits, and you may experience additional charges based on your use of other services such as Amazon Elastic Compute Cloud (Amazon EC2).
