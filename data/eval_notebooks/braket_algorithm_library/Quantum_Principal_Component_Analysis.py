#!/usr/bin/env python
# coding: utf-8

# $$\newcommand{\ket}[1]{\left|{#1}\right\rangle}$$
# $$\newcommand{\bra}[1]{\left\langle{#1}\right|}$$
# $$\newcommand{\braket}[2]{\left\langle{#1}\middle|{#2}\right\rangle}$$

# # Quantum Principal Component Analysis

# Recent areas of computing such as Machine Learning had to face different obstacles to achieve their successes. An example of these problems is the dimensionality of the data that must be processed in our current digital age. With a highly connected world, processing data with high dimensionality is a difficult task due to the resource demand, even for existing supercomputers. For this reason, different techniques have been created to subtract the most relevant information from data sets with high dimensionality, such as the **Principal Component Analysis** (PCA) technique. In a nutshell, PCA seeks to reduce the dimensionality of a data set, while preserving as much variability (i.e., statistical information) as possible [1].

# On the other hand, all these techniques to improve data processing and extract the most relevant information have been conceived under the paradigm of classical computing. Classical computing receives its name to differentiate it from the paradigm of quantum computing which leverages the laws of quantum mechanics to take advantage of new properties that are not present in classical computing [2]. Therefore, these and other machine learning techniques are now being explored using this new paradigm (this new branch is known as Quantum Machine Learning [3]) since these quantum devices can potentially store and process exponentially more information than their classical counterparts. 

# The purpose of this notebook is to implement PCA on a quantum processor by using Amazon Braket. We will use the well-known scenario for PCA of housing prices in the United States (the same one used by [7] from 1996). The idea behind this scenario is to show how a single variable (i.e. price of the house) can be function of many other features such as the number of bedrooms, number of bathrooms, if the kitchen has been recently remodeled, if it has a balcony, square footage, if it has sea view, lot size, amount of years of the construction, location of the house (neighborhood), if it has attic or basement, number of parking spots, etc. Due to the amount of features, it is important to reduce the data to the features that capture the largest variance in the data. Therefore, we are going to use the PCA technique which seeks to determine which features capture the largest variance on the data [1].

# For our case, let's consider the number of bedrooms (first feature) and the square footage (second feature) of several houses for sale in Los Alamos (following the same example as [4]). Here is the raw data, taken from __[Zillow](http://www.zillow.com)__, for 15 houses:

# Number of bedrooms:
# \begin{align}
# X_1 = [4,3,4,4,3,3,3,3,4,4,4,5,4,3,4]
# \end{align}

# Square footage:
# 
# \begin{align}
# X_2 = [3028, 1365, 2726, 2538, 1318, 1693, 1412, 1632, 2875, 3564, 4412, 4444, 4278, 3064, 3857]
# \end{align}

# ## Principal Component Analysis

# PCA has three main steps. The first one is called standarization which means we must adjust the data to an appropiate scale (the idea is to range the continuous initial variables so that each one of them contributes equally to the analysis. To learn more about this point you can see the following [link](https://erdogant.github.io/pca/pages/html/Algorithm.html). Then, we're going to divide each feature ($X_1$,$X_2$) by the standard deviation ($\sigma$) so we can have both features on the same scale, and then we're going to substract off the mean of both features (we substract the mean because we want the features to have the features of a standard normal distribution with mean of zero): 

# In[6]:


import numpy as np

X_1 = [4, 3, 4, 4, 3, 3, 3, 3, 4, 4, 4, 5, 4, 3, 4]
X_2 = [3028, 1365, 2726, 2538, 1318, 1693, 1412, 1632, 2875, 3564, 4412, 4444, 4278, 3064, 3857]
X_1 = (X_1 - np.average(X_1)) / np.std(X_1)
X_2 = (X_2 - np.average(X_2)) / np.std(X_2)


# In[7]:


print("The rescaled feature vectors are")
print("X_1 = ", X_1)
print("X_2 = ", X_2)


# The next step for PCA is calculating the covariance matrix (variance is a measure of dispersion and can be defined as the spread of data from the mean of the given dataset. Covariance is calculated between two variables and is used to measure how the two variables vary together. To learn more about these terms you can see the following [link](https://www.cuemath.com/algebra/covariance-matrix/)) which is defined by (where $\sigma[X]$ is the standard deviation of $X$, and note the $cov[X,X]=\sigma^2[X]=var(X)$ which means the covariance of variable with itself is the variance of the variable): 
# 
# \begin{align}
# \Sigma = 
# \begin{pmatrix}
# var[X_1] & cov[X_1 * X_2] \\
# cov[X_2 * X_1] & var[X_2]
# \end{pmatrix}
# \end{align}

# We're going to use pandas library on Python to calculate the covariance matrix we need:

# In[8]:


import pandas as pd

df = pd.DataFrame({"X_1": X_1, "X_2": X_2})


# In[9]:


sigma = df.cov()
print(sigma)


# Then, our covariance matrix is:
# 
# \begin{align}
# \Sigma = 
# \begin{pmatrix}
# 1.071429 & 0.874147 \\
# 0.874147 & 1.071429
# \end{pmatrix}
# \end{align}

# Now, the final step is to find the eigenvalues and the eigenvectors of the covariance matrix so we can find the Principal Components of the data. According to the number of the eigenvalues we know how much variance is being captured by each principal component. In our 2-feature case, we would keep the component with the highest eigenvalue. We can calculate using classical computation these eigenvalues with the linalg library on Python of Numpy (remember that we are going to diagonalize the covariance matrix to find the principal components of our technique):

# In[10]:


sigma_eigenvalues, sigma_eigenvectors = np.linalg.eig(sigma)
print("sigma_eigenvalues: ", sigma_eigenvalues)
print("sigma_eigenvectors: ", sigma_eigenvectors)


# Then, our eigenvalues (calculated with classical computation) are:
# 
# \begin{align}
# e_1 = 1.945575, e_2 = 0.197281
# \end{align}

# The reason we are calculating the eigenvalues and eigenvectors is because the principal components are eigenvectors of the data's covariance matrix (this can be mathematically shown but it is out of the scope of this notebook. To learn more information related to eigenvectors and eigenvalues on PCA see [this blog](https://medium.com/@dareyadewumi650/understanding-the-role-of-eigenvectors-and-eigenvalues-in-pca-dimensionality-reduction-10186dad0c5c).

# ## Quantum algorithm to compute PCA

# We're going to implement the same quantum algorithm proposed by [4] which has the following steps:
# 
#     1. Classical pre-processing.
#     2. State preparation.
#     3. Purity calculation.
#     4. Classical post-processing.

# ### 1. Classical pre-processing

# Before talking about the classical pre-processing step, it's important to understand what a density matrix is. Density matrices are used to represent quantum states when they're not pure states but mixed states. If we don't know the exact state where our quantum system is but rather we know that it can be in one of the $M$ states, $\ket{\psi_i}$, each with a probability of $p_i$. Then, we can define the density matrix for the quantum system to be: 
# 
# \begin{align}
# \rho = \sum_{i=1}^{M} p_i \ket{\psi_i} \bra{\psi_i}
# \end{align}

# Note that the previous density matrix definition gives the same result when we know the quantum system is in a specific state with $p=1$. Now, it's important to note also that from the desnity matrix definition it can be seen that: 
# 
#     i) it's positive semi-definite
#     ii) it has unit trace
#     
# In fact, **any matrix that satisfies these two properties can be interpreted as a density matrix** [4] (more details about density matrix interpretation on the reference). This important fact is how we are going to encode the classical information of the covariance matrix into a quantum state. 

# Then, our density matrix from our covariance matrix in this case of two features would be (after normalization): 
# 
# \begin{align}
# \rho = \frac{\Sigma}{Tr(\Sigma)}
# \end{align}

# In[11]:


rho = sigma / np.trace(sigma)
print(np.array(rho))


# Our density matrix $\rho$ would be:
# 
# \begin{align}
# \rho = 
# \begin{pmatrix}
# 0.5 & 0.407935 \\
# 0.407935 & 0.5
# \end{pmatrix}
# \end{align}

# ## 2. State preparation

# Before the state preparation step, we must talk about the purification process. Density matrices are a more useful way to represent quantum systems since we can have both pure and mixed states, whereas state vectors can only represent pure states. However, even a mixed state can be seen as a part of a larger system that is in a pure state. This process of converting a mixed state into a pure state of an enlarged system is called **purification** [4].

# Recalling the definition of the density matrix, and by using the [Schrödinger–HJW theorem](https://en.wikipedia.org/wiki/Schr%C3%B6dinger%E2%80%93HJW_theorem) it can be shown that any density matrix $\rho$ can be purified, which means it can be seen as the partial trace of a pure state defined in a larger Hilbert space. In other words, it's always possible to find a larger Hilbert space $\mathcal{H_a}$ with a pure state $\ket{\psi_{sa}} \in \mathcal{H_s} \otimes \mathcal{H_a}$ such that $\rho = Tr_a(\ket{\psi_{sa}}\bra{\psi_{sa}})$, and those states satisfy (for some orthonormal basis {$a_i$}): 
# 
# \begin{align}
# \ket{\psi_{sa}} = \sum_i \sqrt{p_i} \ket{\phi_i} \otimes \ket{a_i}
# \end{align}

# For the state preparation process, it's important to note that in our case of only two features, $\Sigma$ and $\rho$, are $2x2$ matrices (one qubit states), then $\rho$, can be purified to a pure state $\ket{\psi}$ on two qubits. Rigorously, we should design a state preparation circuit for this purification process which is made on [4] with details but for this notebook we directly calculate it to focus on the quantum PCA process.

# First, we find the eigenvalues and eigenvector of our density matrix $\rho$. 

# In[12]:


rho_eig_val, rho_eig_vec = np.linalg.eig(rho)
print("rho_eig_val: ", rho_eig_val)
print("rho_eig_vec: ", rho_eig_vec)


# Then, our eigenvectors (after normalization process) would be:
# 
# \begin{align}
# \vec{v_1} = 
# \begin{pmatrix}
# 0.707106 \\
# -0.707106
# \end{pmatrix},
# \vec{v_2} = 
# \begin{pmatrix}
# -0.707106 \\
# -0.707106
# \end{pmatrix}
# \end{align}
# 

# Following the definition of the pure state, we are going to use the computational basis which is an orthonormal basis (i.e. {$a=\ket0,\ket1$}):

# In[13]:


sqrt_eig_val = np.sqrt(rho_eig_val)
print(sqrt_eig_val)


# Our eigenvalues would be:
# 
# \begin{align}
# \sqrt{\lambda_1} = 0.303421, \sqrt{\lambda_2} = 0.952856
# \end{align}

# Using the previous definition, our pure state would be: 
# 
# \begin{align}
# \ket{\psi} = \sum_i \sqrt{\lambda_i} \ket{v_i} \otimes \ket{a_i}
# \end{align}

# \begin{align}
# \ket{\psi} = \sqrt{\lambda_1} \ket{v_1} \otimes \ket{0} + \sqrt{\lambda_2} \ket{v_2} \otimes \ket{1}
# \end{align}

# \begin{align}
# \ket{\psi} = 0.303421
# \begin{pmatrix}
# 0.707106 \\
# -0.707106
# \end{pmatrix} \otimes 
# \begin{pmatrix}
# 1 \\
# 0
# \end{pmatrix} + 0.952856
# \begin{pmatrix}
# -0.707106 \\
# -0.707106
# \end{pmatrix} \otimes 
# \begin{pmatrix}
# 0 \\
# 1
# \end{pmatrix}
# \end{align}

# \begin{align}
# \ket{\psi} = 0.303421
# \begin{pmatrix}
# 0.707106 \\
# 0 \\
# -0.707106  \\
# 0
# \end{pmatrix} + 0.952856
# \begin{pmatrix}
# 0 \\
# -0.707106 \\
# 0  \\
# −0.707106
# \end{pmatrix}
# \end{align}

# We must use the *flip* function of Numpy because the eigenvectors provided by the linalg library are actually the columns and not the rows:

# In[14]:


tensor_product1 = np.vstack((np.flip(rho_eig_vec[1]), np.zeros(2))).ravel("F")
tensor_product2 = np.vstack((np.zeros(2), np.flip(rho_eig_vec[0]))).ravel("F")
print(tensor_product1)
print(tensor_product2)


# In[15]:


psi = sqrt_eig_val[0] * tensor_product1 + sqrt_eig_val[1] * tensor_product2
print(psi)


# Our purified state would be: 
# 
# \begin{align}
# \ket{\psi} = 
# \begin{pmatrix}
# -0.214551 & -0.673771 \\
# 0.214551 & -0.673771 \\
# \end{pmatrix}
# \end{align}

# The way to confirm the state above is purified is checking if we can recover the original density matrix $\rho$. 
# 
# \begin{align}
# \rho = Tr_a(\ket{\psi}\bra{\psi}) 
# \end{align}

# In[16]:


rho_partial_trace = np.dot(psi.reshape((4, 1)), psi.reshape((4, 1)).transpose())
print(rho_partial_trace)


# As the reader can verify, taking the partial trace of the previous matrix lead us to the original mixed state with density matrix $\rho$ which proves that $\ket{\psi}$ is a purified state. 
# 
# \begin{align}
# Tr_a(\ket{\psi}\bra{\psi}) = \begin{pmatrix}
# 0.5 & 0.407935 \\
# 0.407935 & 0.5
# \end{pmatrix}
# = \rho
# \end{align}

# ## 3. Purity calculation

# Before getting into the details of calculating the purity, we must discuss what the purity is and how to find it in our scenario. The purity is a measure which states how much a quantum state is mixed and is defined by: 
# 
# \begin{align}
# P = Tr(\rho^2) = \sum_{i} p_i^2
# \end{align}

# Using the useful definitions and theory provided by [5], we can see that the purity can be expressed in terms of the lenght $r = || \vec{r} ||$ of the **Bloch vector**: 
# 
# \begin{align}
# P = Tr(\rho^2) = \frac{1 + r^2}{2}
# \end{align}
# 
# \begin{align}
# 2P - 1 = r^2
# \end{align}
# 
# \begin{align}
# \sqrt{2P - 1} = || \vec{r} ||
# \end{align}

# Using the definition of the eigenvalues of the density matrix $\rho$ expressed in terms of the lenght $r$ of the Bloch vector:
# 
# \begin{align}
# \lambda_{1,2} =  \frac{(1 \pm \sqrt{(x^2 + y^2 + z^2)})}{2} = \frac{(1 \pm ||\vec{r} ||)}{2} = \frac{(1 \pm \sqrt{2P-1})}{2}
# \end{align}

# The previous equations states that we can calculate the eigenvalues of our covariance matrix (i.e. the eigenvalues for the Principal Components) by measuring the purity of our quantum state. 

# Now, following the same procedure presented by [6], the purity is equal to the expected value of the SWAP gate between two purification copies (using also the previous definitions):

# \begin{align}
# \rho = \sum_{i} p_i \ket{v_i} \bra{v_i}
# \end{align}
# 
# \begin{align}
# \ket{\psi} = \sum_{i} \sqrt{p_i} \ket{v_i} \otimes \ket{w_i} = \sum_{i} \sqrt{p_i} \ket{v_i} \ket{w_i}
# \end{align}
# 
# \begin{align}
# P = Tr(\rho^2) = \sum_{i} p_i^2
# \end{align}
# 
# 

# Now, let's see what the SWAP operation makes to the purified state:
# 
# \begin{align}
# \bra{\psi} \bra{\psi} SWAP_{a} \ket{\psi} \ket{\psi} = 
# \sum_{ij} \bra{w_{j}} \bra{v_{j}} \bra{w_{i}}  \bra{v_{i}} \sqrt{p_i} \sqrt{p_j} \ket{v_{j}} \ket{w_{i}} \ket{v_{i}} \ket{w_{j}} = 
# \sum_{i} p_i^2
# \end{align}

# Then, 
# 
# \begin{align}
# P = Tr(\rho^2) = \bra{\psi} \bra{\psi} SWAP_{a} \ket{\psi} \ket{\psi}
# \end{align}

# The previous equation states that purity can be measured using the Hadamard test on a controlled-SWAP gate. We're going to use the Amazon Braket service to find the purity of our quantum system to finally convert that data into the eigenvalues we're looking for our PCA scenario.
# 
# To do this, we're going to follow the algorithm proposed by [4] which involves two processes: the state preparation and the purity calculation. 

# In[17]:


# AWS imports: Import Braket SDK modules
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.circuits import Gate
from braket.circuits import Observable
import matplotlib.pyplot as plt
from braket.aws import AwsDevice


# ## Implementation on Amazon Braket

# We just need to create our quantum circuit with 5 qubits (two copies for the quantum state which means we need 4 qubits and a fifth qubit to interact with the answer which is known as an ancilla qubit). Each pair of qubits must be initialized to start with the quantum state that we already found on the previous part of this notebook (i.e. $\ket{\psi}$ and the last qubit just needs to be initialized as the $\ket{0}$ state. Then, we need to apply a Hadamard gate to the ancilla qubit so we can use the superposition property before applying the controlled-SWAP gate between the ancilla qubit and the pair of quantum states (here the basis state is the target qubit and the control qubits are the two copies of the quantum state). Finally, another Hadamard gate to recover the answer we're looking for and the measure over the ancilla qubit. 
# 
# Then, we need to implement each step using the Amazon Braket SDK because not all quantum providers offer the same gates, that's why we need to decompose complex gates (e.g. Controlled-SWAP gate) into simple gates that any quantum hardware provider can run. That is the reason why we are using the Amazon Braket SDK since it allows us to write one circuit that can be compiled down to run on various devices.

# Let's start by defining the quantum gate that we'll use many times during our quantum circuit: the unitary gate. This unitary gate is a generic single-qubit rotation gate with 3 Euler angles defined by:

# \begin{align}
# U(\theta, \phi, \lambda) = 
# \begin{pmatrix}
# cos(\frac{\theta}{2}) & -e^{i\lambda}sin(\frac{\theta}{2}) \\
# e^{i\phi}sin(\frac{\theta}{2}) & e^{i(\phi+\lambda)}cos(\frac{\theta}{2}) \\
# \end{pmatrix}
# \end{align}

# In[18]:


def rotation_matrix(value):
    return np.array(
        [
            [np.cos(value / 2), -np.exp(1j * value) * np.sin(value / 2)],
            [np.exp(1j * value) * np.sin(value / 2), np.exp(2 * 1j * value) * np.cos(value / 2)],
        ]
    )


# Considering our special case has only 2 features, for the PCA process we are doing the following values can be calculated using the above equations and data. First, we need to take the basis states to the quantum state $\ket{\psi}$ we found earlier (this is the state preparation process using unitary matrices, hadamard and cnot gates). Then, for the purity calculation process we need to decompose the controlled-swap operation into simple gates (the way of doing this is beyond the scope of this notebook, but if the reader wants further details about how to do decomposition using the Toffoli gate please see the references provided by [4] on this topic). Finally, we'd use the same strategy of the ancilla qubit to measure the quantum circuit to get the eigenvalues we're looking for.

# In[19]:


circuit = Circuit()

for i in [1, 2]:
    circuit.unitary(matrix=rotation_matrix(0.465), targets=[i])

circuit.h(3)

for i, j in [[1, 0], [2, 4]]:
    circuit.cnot(i, j)

for i in [0, 4]:
    circuit.unitary(matrix=rotation_matrix(1.570), targets=[i])

for i in [1, 2]:
    circuit.unitary(matrix=rotation_matrix(1.950), targets=[i])

for i in [1, 2]:
    circuit.h(i)

circuit.cnot(2, 1)

circuit.h(2)

circuit.cnot(2, 1)

circuit.ti(1)

for i, j in [[2, 1], [3, 2], [2, 1]]:
    circuit.cnot(i, j)

circuit.t(1)

for i, j in [[3, 2], [2, 1]]:
    circuit.cnot(i, j)

circuit.ti(1)

for i, j in [[2, 1], [3, 2], [2, 1]]:
    circuit.cnot(i, j)

for i, j, k in [[1, 3, 2], [2, 3, 2]]:
    circuit.t(i)
    circuit.cnot(j, k)

circuit.ti(2)
circuit.t(3)

circuit.cnot(3, 2)

for i in [2, 3]:
    circuit.h(i)

circuit.cnot(2, 1)

# measurement part of the quantum circuit
circuit.expectation(Observable.Z(), target=[3])
circuit.sample(observable=Observable.Z(), target=3)
circuit.probability(target=3)

print(circuit)


# ## Amazon Braket local simulator

# For the full article, please go to the following [link](https://aws.amazon.com/es/blogs/quantum-computing/simulating-quantum-circuits-with-amazon-braket/).
# 
# Today’s quantum computers still have limited capacity, qubit counts, and accuracy. Therefore, quantum circuit simulators, which emulate the behavior of quantum computers using classical hardware, are often the tool of choice to study quantum algorithms for quantum computing researchers and enthusiasts. Amazon Braket provides a suite of quantum simulators for different use cases to help customers prototype and develop quantum algorithms [8].
# 
# The most common use case for simulators is to prototype, validate, and debug quantum programs. Whether you want to understand the foundation of Simon’s algorithm, or validate that your variational algorithm runs without a bug before running it on a quantum computer, it is useful to have a quick way to run small-scale circuits. For this reason, the Amazon Braket SDK comes pre-installed with a local simulator that runs wherever you use the SDK, for example, on Amazon Braket notebooks, or your own laptop [8].

# In[20]:


local_device = LocalSimulator()

local_result = local_device.run(circuit, shots=10000).result()
local_counts = local_result.measurement_counts
print(local_counts)


# In[21]:


print("Probability:", local_result.values[2])


# In[22]:


purity_sdk = local_result.values[2][0] - local_result.values[2][1]
print(purity_sdk)


# In[23]:


v_1 = (1 + np.sqrt(2 * purity_sdk - 1)) / 2 * np.trace(sigma)
print("The first eigenvalue obtained by the quantum PCA using Amazon Braket SDK is: \n", v_1)


# Recall the eigenvalues we found using linear algebra at the beginning of this notebook were:
# 
# \begin{align}
# e_1 = 1.945575, e_2 = 0.197281
# \end{align}

# In[33]:


perc_v1 = abs((v_1 - 1.945575) / 1.945575) * 100
print("percent error first eigenvalue (%): ", perc_v1)


# ## Amazon Braket SV1 device

# State vector simulators are the workhorse of circuit simulation. They keep a precise representation of the quantum state (the state vector) at every point in the simulation, and iteratively change the state under the action of the different gates of the circuit, one by one. Each gate that is applied corresponds to a matrix-vector multiplication resulting in predictable runtimes of SV1 with little variation [8].

# In[25]:


sv1_device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

sv1_result = sv1_device.run(circuit, shots=10000).result()
sv1_counts = sv1_result.measurement_counts
print(sv1_counts)


# In[26]:


print("Expectation value for:", sv1_result.values[2])


# In[27]:


purity_sv1 = sv1_result.values[2][0] - sv1_result.values[2][1]
print(purity_sv1)


# In[28]:


a_1 = (1 + np.sqrt(2 * purity_sv1 - 1)) / 2 * np.trace(sigma)
print("The first eigenvalue obtained by the quantum PCA using Amazon Braket SV1 is: \n", a_1)


# Recall the eigenvalues we found using linear algebra at the beginning of this notebook were:
# 
# \begin{align}
# e_1 = 1.945575, e_2 = 0.197281
# \end{align}

# In[32]:


perc_sv1 = abs((a_1 - 1.945575) / 1.945575) * 100
print("percent error first eigenvalue (%): ", perc_sv1)


# # Conclusions

# Summarizing what we've done, along the entire notebook we implemented the Principal Component Analysis technique under the quantum computation paradigm. We implemented the PCA technique for the famous problem of house pricing in the US where our final goal is to find the eigenvalues of our correlation matrix of our data (i.e. the principal components). The way how we implemented the PCA technique was divided in four stages: i) Classical pre-processing, ii) State preparation, iii) Purity calculation, iv) Classical post-processing. After explaining each step with the mathematics and physics details needed, we implemented the quantum algorithm using the Amazon Braket SDK with two devices: the local simulator and the Amazon Braket SV1. 
# 
# Our results for the eigenvalue we were looking for is the same we predicted using classical computation with a percent error less than 5%. Finally, we saw that for this particular scenario, PCA technique was correctly implemented using quantum algorithms, circuits and devices. However, we know the technology is still in an early stage, but proving that we can make these small steps brings hope and optimism for the future of quantum computing and its applications on machine learning. 

# # References
# 
# 1. Ian T. Jolliffe and Jorge Cadima. Principal component analysis: A review and recent developments. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 374, April 2016.
# 
# 2. C. He, J. Li, W. Liu, J. Peng, and Z. J. Wang. A low-complexity quantum principal component analysis algorithm,. IEEE Transactions on Quantum Engineering, 3(3):1–13, 2022.
# 
# 3. Lin Jie et al. An improved quantum principal component analysis algorithm based on the quantum singular threshold method. Physics Letters A, 383:2862–68, aug 2019.
# 
# 4. J. , Abhijith, et al. «Quantum Algorithm Implementations for Beginners». ACM Transactions on Quantum Computing, vol. 3, n.o 4, dic 2022, pp. 1-92. arXiv.org, https://doi.org/10.1145/3517340.
# 
# 5. Schmied, Roman. «Quantum State Tomography of a Single Qubit: Comparison of Methods». Journal of Modern Optics, vol. 63, n.o 18, oct 2016, pp. 1744-58. DOI.org (Crossref), https://doi.org/10.1080/09500340.2016.1142018.
# 
# 6. An special thanks to the Github repository of Haokai Zhang available at: https://github.com/Haokai-Zhang/ExampleQPCA/blob/master/5qubit-qPCA.ipynb
# 
# 7. Kelley Pace, R., y Ronald Barry. «Sparse spatial autoregressions». Statistics & Probability Letters, vol. 33, n.o 3, may 1997, pp. 291-97. ScienceDirect, https://doi.org/10.1016/S0167-7152(96)00140-X.
# 
# 8. Simulating Quantum Circuits with Amazon Braket | AWS Quantum Technologies Blog. sept 2021, https://aws.amazon.com/blogs/quantum-computing/simulating-quantum-circuits-with-amazon-braket/.
# 
# 
