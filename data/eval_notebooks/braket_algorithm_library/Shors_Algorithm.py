#!/usr/bin/env python
# coding: utf-8

# # Shor's Algorithm
# 
# This example provides an implementation of the Shor's algorithm using the Amazon Braket SDK. Shor's algorithm is used to find prime factors of an integer. On a quantum computer, Shor's algorithm runs in polynomial time and is almost exponentially faster than the most efficient known classical factoring algorithm. The efficiency of Shor's algorithm is due to the efficiency of the Quantum Fourier transform, Quantum Phase estimation and modular exponentiation by repeated squarings. In this notebook, you implement the Shor's algorithm in code using the Amazon Braket SDK and run a simple example of factoring 15.

# # References 
# 
# [1] Wikipedia: https://en.wikipedia.org/wiki/Shor%27s_algorithm
# 
# [2] Nielsen, Michael A., Chuang, Isaac L. (2010). Quantum Computation and Quantum Information (2nd ed.). Cambridge: Cambridge University Press.

# In[1]:


from braket.devices import LocalSimulator
from braket.aws import AwsDevice
from braket.experimental.algorithms.shors.shors import (
    shors_algorithm,
    run_shors_algorithm,
    get_factors_from_results,
)


# # Prepare inputs for Shor's Algorithm

# In[2]:


N = 15  # Integer to factor (currently 15, 21, 35 work)
a = 7  # Any integer that satisfies 1 < a < N and gcd(a, N) = 1.


shors_circuit = shors_algorithm(N, a)


# # Run on a local simulator

# In[3]:


local_simulator = LocalSimulator()

output = run_shors_algorithm(shors_circuit, local_simulator)

guessed_factors = get_factors_from_results(output, N, a)


# # Run on a managed simulator
# 
# 

# In[4]:


# Use Braket SDK Cost Tracking to estimate the cost to run this example
from braket.tracking import Tracker

tracker = Tracker().start()


# In[5]:


managed_sim = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
output = run_shors_algorithm(shors_circuit, managed_sim)

guessed_factors = get_factors_from_results(output, N, a)


# In[6]:


print("Task Summary")
print(f"{tracker.quantum_tasks_statistics()} \n")
print(
    f"Estimated cost to run this example: {tracker.qpu_tasks_cost() + tracker.simulator_tasks_cost():.2f} USD"
)


# Note: Charges shown are estimates based on your Amazon Braket simulator and quantum processing unit (QPU) task usage. Estimated charges shown may differ from your actual charges. Estimated charges do not factor in any discounts or credits, and you may experience additional charges based on your use of other services such as Amazon Elastic Compute Cloud (Amazon EC2).
