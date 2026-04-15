#!/usr/bin/env python
# coding: utf-8

# # Bell's Inequality
# 
# This tutorial shows how to run a version of Bell's inequality experiment in Braket on a local simulator and a QPU.   

# ## References 
# 
# [1]  Bell, J. S. On the Einstein Podolsky Rosen Paradox. Physics Physique Fizika 1, no. 3 (November 1, 1964): 195–200. https://doi.org/10.1103/PhysicsPhysiqueFizika.1.195. 
# 
# [2] Greenberger, Daniel M., Michael A. Horne, Abner Shimony, and Anton Zeilinger (1990). Bell’s Theorem without Inequalities. American Journal of Physics 58, no. 12: 1131–43. https://doi.org/10.1119/1.16243. 

# # Run on a local simulator

# In[1]:


import numpy as np
from braket.devices import LocalSimulator
from braket.tracking import Tracker

from braket.experimental.algorithms.bells_inequality import (
    create_bell_inequality_circuits,
    get_bell_inequality_results,
    run_bell_inequality,
)

tracker = Tracker().start()  # to keep track of Braket costs


# Bell's Inequality experiment consists of three circuits acting on two qubits each. 

# In[2]:


circAB, circAC, circBC = create_bell_inequality_circuits(0, 1)
print("\nCircuit AB\n", circAB)
print("\nCircuit AC\n", circAC)
print("\nCircuit BC\n", circBC)


# The three circuits are grouped together in the `run_bell_inequality` function below. To run on a local noise-free simulator, we can call this function. 

# In[3]:


local_simulator = LocalSimulator()
local_tasks = run_bell_inequality([circAB, circAC, circBC], local_simulator, shots=1000)
print(local_tasks)


# To gather the results of Bell's inequality test, we call the `get_bell_inequality_results` of the tasks from above. 

# In[4]:


results, pAB, pAC, pBC = get_bell_inequality_results(local_tasks)


# # Run on a QPU 
# 
# To run Bell's inequality on a QPU, we replace the LocalSimulator with an AwsDevice. The cost to run this experiment is \\$0.3 per task and \\$0.00145 per shot on the IQM Garnet device. Since we have three circuits of 1000 shots each, that totals \$5.25 USD.

# In[5]:


# # Uncomment to run on a QPU
# from braket.aws import AwsDevice
# iqm_garnet = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
# iqm_tasks = run_bell_inequality([circAB, circAC, circBC], iqm_garnet, shots=1000)
# results, pAB, pAC, pBC = get_bell_inequality_results(iqm_tasks)


# We see that Bell's inequality is violated, so the device is demonstrating quantum behavior.

# In[6]:


print("Task Summary")
print(f"{tracker.quantum_tasks_statistics()} \n")
print(
    f"Estimated cost to run this example: {tracker.qpu_tasks_cost() + tracker.simulator_tasks_cost():.2f} USD"
)


# Note: Charges shown are estimates based on your Amazon Braket simulator and quantum processing unit (QPU) task usage. Estimated charges shown may differ from your actual charges. Estimated charges do not factor in any discounts or credits, and you may experience additional charges based on your use of other services such as Amazon Elastic Compute Cloud (Amazon EC2).
