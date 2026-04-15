#!/usr/bin/env python
# coding: utf-8

# # CHSH Inequality
# 
# This tutorial shows how to run the CHSH (Clauser, Horne, Shimony and Holt) inequality experiment in Braket on local simulator and a QPU.
# 
# For easy understanding, variables used here follow closely the reference: Scientific Background on the Nobel Prize in Physics 2022 [1]
# 
# 
# # Background 
# 
# The CHSH inequality is a generalization of Bell's inequality [2]. For a singlet state
# $$|\psi ^{-}\rangle = \frac{1}{\sqrt{2}}\left(|+-\rangle + |-+\rangle \right )$$
# the CHSH inequality is
# 
# $$
# |E(A_1,B_1) + E(A_1,B_2) + E(A_2,B_1) - E(A_2,B_2)| \leq 2
# $$
# for measurement settings $A_1, A_2$ on the first particle and settings $B_1,B_2$ on the second particle. This reduces to Bell's original inequality if $A_1=B_1$.

# ## References 
# 
# [1] The Nobel Committee for Physics, Scientific Background on the Nobel Prize in Physics 2022, https://www.nobelprize.org/uploads/2022/10/advanced-physicsprize2022.pdf
# 
# [2] John F. Clauser, Michael A. Horne, Abner Shimony, and Richard A. Holt. Proposed Experiment to Test Local Hidden-Variable Theories. Phys. Rev. Lett. 23, 880 – Published 13 October 1969; Erratum Phys. Rev. Lett. 24, 549 (1970) https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.23.880

# # Run on a local simulator

# In[1]:


import numpy as np
from braket.devices import LocalSimulator
from braket.tracking import Tracker

from braket.experimental.algorithms.chsh_inequality import (
    create_chsh_inequality_circuits,
    run_chsh_inequality,
    get_chsh_results,
)

tracker = Tracker().start()  # to keep track of Braket costs


# CHSH Inequality experiment consists of four circuits acting on two qubits each. The four circuits are grouped together in the `create_chsh_inequality_circuits` function. The input arguments are the two qubits to act on and the two separate angles to rotate each qubits by. The default values for the angles are:
# $$A_1 = \pi/2$$
# $$B_1 = \pi/4$$
# $$A_2 = 0$$
# $$B_2 = 3\pi/4$$
# which gives maximum violation of the CHSH inequality. 
# 
# Below, we print the four CHSH circuits.

# In[2]:


circuits = create_chsh_inequality_circuits(a1=np.pi / 2, a2=0, b1=np.pi / 4, b2=3 * np.pi / 4)

print(
    "\ncircuit_a1b1: measurement setting for qubit0 at angle a1, qubit1 at angle b1\n", circuits[0]
)
print(
    "\ncircuit_a1b2: measurement setting for qubit0 at angle a1, qubit1 at angle b2\n", circuits[1]
)
print(
    "\ncircuit_a2b1: measurement setting for qubit0 at angle a2, qubit1 at angle b1\n", circuits[2]
)
print(
    "\ncircuit_a2b2: measurement setting for qubit0 at angle a2, qubit1 at angle b2\n", circuits[3]
)


# The circuits can be run on the Braket local simulator with:

# In[3]:


local_tasks = run_chsh_inequality(circuits, LocalSimulator(), shots=0)


# The results of the inequality experiment are called using the `get_chsh_results` function below.

# In[4]:


chsh_value, results, E_a1b1, E_a1b2, E_a2b1, E_a2b2 = get_chsh_results(local_tasks, verbose=True)


# Notice the CHSH value is very close to $2\sqrt{2}\approx 2.828$. 

# Below, we plot the CHSH value for various values of the angles. 

# In[5]:


local_simulator = LocalSimulator()
angles = np.linspace(0, 2 * np.pi, 100)

chsh_inequality_lhs_max = 0
chsh_inequality_lhs_max_theta = 0
chsh_values = []

for theta in angles:
    circuits = create_chsh_inequality_circuits(a2=0, b1=theta, a1=2 * theta, b2=3 * theta)
    local_tasks = run_chsh_inequality(circuits, local_simulator, shots=0)
    chsh_value, results, E_a1b1, E_a1b2, E_a2b1, E_a2b2 = get_chsh_results(
        local_tasks, verbose=False
    )
    if chsh_value > chsh_inequality_lhs_max:
        chsh_inequality_lhs_max = np.abs(chsh_value)
        chsh_inequality_lhs_max_theta = theta
    chsh_values.append(chsh_value)

print(
    "\nFor all the iterations:\n Max CHSH_inequality:",
    chsh_inequality_lhs_max,
    "Corresponding theta:",
    chsh_inequality_lhs_max_theta,
)


# Plotting the CHSH Value against theta angle to determine which theta gives maximum violation.

# In[6]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(angles, chsh_values, ".", label="CHSH Value")

plt.grid(which="major", axis="both")
plt.yticks(
    [-np.sqrt(2) * 2, -2, 0, 2, np.sqrt(2) * 2],
    ["$-2\sqrt{2}$", "$-2$", "$0$", "$2$", "$2\sqrt{2}$"],
)
plt.xticks(
    [i * np.pi / 4 for i in range(9)],
    [
        "$0$",
        "$\pi/4$",
        "$\pi/2$",
        "$3\pi/4$",
        "$\pi$",
        "$5\pi/4$",
        "$3\pi/2$",
        "$7\pi/4$",
        "$2\pi$",
    ],
)

plt.xlabel("Angle")
plt.ylabel("CHSH Value");


# From the plot, we see that the maximum violation is for $\theta = \pi/4$ as expected. The red line shows the classical bounds of $\text{CHSH} \leq 2$ and the dotted black line shows the quantum bound at $\text{CHSH} \leq 2\sqrt{2}$. 
# 
# 
# Following sections how CHSH reduces to Bell's original inequality. 

# In[7]:


# CHSH reduces to Bell's original inequality measurement configuration with a2=b1 and b2 middle of a1 and b1
# thumb rule: left hand index finger within the right hand thumb and index finger.
# left hand thumb aligned with righ hand index finger

local_simulator = LocalSimulator()
angles = np.linspace(0, 2 * np.pi, 100)

bell_value_max = 0
bell_value_max_theta = 0
bell_value_max_E_a1b1 = 0
bell_value_max_E_a1b2 = 0
bell_value_max_E_a2b1 = 0
bell_value_max_E_a2b2 = 0
bell_values = []

for theta in angles:
    circuits = create_chsh_inequality_circuits(a2=0, b1=0, b2=theta, a1=2 * theta)
    local_tasks = run_chsh_inequality(circuits, local_simulator, shots=0)
    chsh_value, results, E_a1b1, E_a1b2, E_a2b1, E_a2b2 = get_chsh_results(
        local_tasks, verbose=False
    )
    # since d=b, E_a2b1 should be -1 and abs(E_a2b1)=1, subtracting this value from both sides of the CSHS inequality gives Bell inequality
    bell_value = np.abs(E_a1b1 - E_a1b2) - E_a2b2
    if bell_value > bell_value_max:
        bell_value_max = np.abs(bell_value)
        bell_value_max_theta = theta
        bell_value_max_E_a1b1 = E_a1b1
        bell_value_max_E_a1b2 = E_a1b2
        bell_value_max_E_a2b1 = E_a2b1
        bell_value_max_E_a2b2 = E_a2b2
    bell_values.append(bell_value)

print(
    "\nFor all the iterations:\nMax Bell_inequality:",
    bell_value_max,
    "\nCorresponding theta:",
    bell_value_max_theta,
    "\nCorresponding E_a1b1, E_a1b2, E_a2b1, E_a2b2:",
    bell_value_max_E_a1b1,
    bell_value_max_E_a1b2,
    bell_value_max_E_a2b1,
    bell_value_max_E_a2b2,
)


# In[8]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(angles, bell_values, ".", label="Bell Inequality Value")

plt.grid(which="major", axis="both")
plt.yticks([-1.5, -1, 0, 1, 1.5], ["$-1.5$", "$-1$", "$0$", "$1$", "$1.5$"])
plt.xticks(
    [i * np.pi / 3 for i in range(7)],
    ["$0$", "$\pi/3$", "$2\pi/3$", "$\pi$", "$4\pi/3$", "$5\pi/3$", "$2\pi$"],
)
plt.xlabel("Angle")
plt.ylabel("Reduction to Bell Inequality Value");


# # Run on a QPU
# 
# To run CHSH inequality on a QPU, we replace the LocalSimulator with an AwsDevice. 
# To reduce the cost, we run the the experiment only the defualt angles which gives the maximum CHSH inequality value.
# 
# The cost to run this experiment is \\$0.3 per task and \\$0.00145 per shot on the IQM Garnet device. Since we have four circuits of 1000 shots each, the total cost is \$7 USD.

# In[9]:


# # Uncomment the following to run on QPU
# from braket.aws import AwsDevice
# device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
# circuits = create_chsh_inequality_circuits()
# tasks = run_chsh_inequality(circuits, device, shots=1_000)
# chsh_value, results, E_a1b1, E_a1b2, E_a2b1, E_a2b2 = get_chsh_results(tasks)


# We see that CHSH inequality is violated, so the device is demonstrating quantum behavior.

# In[10]:


print(
    f"Estimated cost to run this example: {tracker.qpu_tasks_cost() + tracker.simulator_tasks_cost():.2f} USD"
)


# Note: Charges shown are estimates based on your Amazon Braket simulator and quantum processing unit (QPU) task usage. Estimated charges shown may differ from your actual charges. Estimated charges do not factor in any discounts or credits, and you may experience additional charges based on your use of other services such as Amazon Elastic Compute Cloud (Amazon EC2).
