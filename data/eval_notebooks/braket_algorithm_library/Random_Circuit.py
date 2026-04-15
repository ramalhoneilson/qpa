#!/usr/bin/env python
# coding: utf-8

# # Random Circuit
# 
# Generates random quantum circuits using the Amazon Braket SDK.
# 
# ### Circuit Generation for Testing
# 
# Random quantum circuits allow creation of a diverse set of circuits with a variety of output probability distributions. Users can utilize random circuits to test performance of quantum simulators and QPUs. 
# 
# ### Benchmarking quantum compilation stacks
# 
# Random circuits sampled from a fixed gate set (as in the example below) are often used for benchmarking performance of quantum compilation passes, such as circuit mapping, routing, or circuit optimization passes. 

# # Run on a local simulator

# In[1]:


from braket.circuits.gates import CNot, Rx, Rz, CPhaseShift, XY
from braket.devices import LocalSimulator
from braket.experimental.auxiliary_functions import random_circuit

# Code here
local_simulator = LocalSimulator()
gate_set = [CNot, Rx, Rz, CPhaseShift, XY]
circuit = random_circuit(num_qubits=5, num_gates=30, gate_set=gate_set, seed=42)
task = local_simulator.run(circuit, shots=100)
result = task.result()
print("--Circuit--")
print(circuit)
print("\n--Counts--")
print(result.measurement_counts)

