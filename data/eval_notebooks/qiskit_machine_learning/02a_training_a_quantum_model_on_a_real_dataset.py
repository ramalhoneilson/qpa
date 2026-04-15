#!/usr/bin/env python
# coding: utf-8

# # Training a Quantum Model on a Real Dataset
# 
# This tutorial will demonstrate how to train a quantum machine learning model to tackle a classification problem. Previous tutorials have featured small, artificial datasets. Here we will increase the problem complexity by considering a real-life classical dataset. We decided to pick a very well-known – albeit still relatively small – problem: the Iris flower dataset. This dataset even has its own Wikipedia [page](https://en.wikipedia.org/wiki/Iris_flower_data_set). Although the Iris dataset is well known to data scientists, we will briefly introduce it to refresh our memories. For comparison, we'll first train a classical counterpart to the quantum model.
# 
# So, let's get started:
# 
# - First, we'll load the dataset and explore what it looks like.
# - Next, we'll train a classical model using [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) from [scikit-learn](https://scikit-learn.org/) to see how well the classification problem can be solved using classical methods.
# - After that, we'll introduce the Variational Quantum Classifier (VQC).
# - To conclude, we'll compare the results obtained from both models.
# 
# ## 1. Exploratory Data Analysis
# 
# First, let us explore the Iris dataset this tutorial will use and see what it contains. For our convenience, this [dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) is available in scikit-learn and can be loaded easily.

# In[1]:


from sklearn.datasets import load_iris

iris_data = load_iris()


# If no parameters are specified in the `load_iris` function, then a dictionary-like object is returned by scikit-learn. Let's print the description of the dataset and see what is inside.

# In[2]:


print(iris_data.DESCR)


# There are a few interesting observations we can find from this dataset description:
# 
# - There are 150 samples (instances) in the dataset.
# - There are four features (attributes) in each sample.
# - There are three labels (classes) in the dataset.
# - The dataset is perfectly balanced, as there are the same number of samples (50) in each class.
# - We can see features are not normalized, and their value ranges are different, e.g., $[4.3, 7.9]$ and $[0.1, 2.5]$ for sepal length and petal width, respectively. So, transforming the features to the same scale may be helpful.
# - As stated in the table above, feature-to-class correlation in some cases is very high; this may lead us to think that our model should cope well with the dataset.
# 
# We only examined the dataset description, but additional properties are available in the `iris_data` object. Now we are going to work with features and labels from the dataset.

# In[3]:


features = iris_data.data
labels = iris_data.target


# Firstly, we'll normalize the features. Namely, we will apply a simple transformation to represent all features on the same scale. In our case, we squeeze all features onto the interval $[0, 1]$. Normalization is a common technique in machine learning and often leads to better numerical stability and convergence of an algorithm.
# 
# We can use `MinMaxScaler` from scikit-learn to perform this. Without specifying parameters, this does exactly what is required: maps data onto $[0, 1]$.

# In[4]:


from sklearn.preprocessing import MinMaxScaler

features = MinMaxScaler().fit_transform(features)


# Let's see how our data looks. We plot the features pair-wise to see if there's an observable correlation between them.

# In[5]:


import pandas as pd
import seaborn as sns

df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df["class"] = pd.Series(iris_data.target)

sns.pairplot(df, hue="class", palette="tab10")


# From the plots, we see that class `0` is easily separable from the other two classes, while classes `1` and `2` are sometimes intertwined, especially regarding the "sepal width" feature.
# 
# Next, let's see how classical machine learning handles this dataset. 
# 
# ## 2. Training a Classical Machine Learning Model
# 
# Before we train a model, we should split the dataset into two parts: a training dataset and a test dataset. We'll use the former to train the model and the latter to verify how well our models perform on unseen data.
# 
# As usual, we'll ask scikit-learn to do the boring job for us. We'll also fix the seed to ensure the results are reproducible.

# In[6]:


from sklearn.model_selection import train_test_split
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=algorithm_globals.random_seed
)


# We train a classical Support Vector Classifier from scikit-learn. For the sake of simplicity, we don't tweak any parameters and rely on the default values.

# In[7]:


from sklearn.svm import SVC

svc = SVC()
_ = svc.fit(train_features, train_labels)  # suppress printing the return value


# Now we check out how well our classical model performs. We will analyze the scores in the conclusion section.

# In[8]:


train_score_c4 = svc.score(train_features, train_labels)
test_score_c4 = svc.score(test_features, test_labels)

print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")


# As can be seen from the scores, the classical SVC algorithm performs very well. Next up, it's time to look at quantum machine learning models.
# 
# ## 3. Training a Quantum Machine Learning Model
# 
# As an example of a quantum model, we'll train a variational quantum classifier (VQC). The VQC is the simplest classifier available in Qiskit Machine Learning and is a good starting point for newcomers to quantum machine learning who have a background in classical machine learning.
# 
# But before we train a model, let's examine what comprises the `VQC` class. Two of its central elements are the feature map and ansatz. What these are will now be explained.
# 
# Our data is classical, meaning it consists of a set of bits, not qubits. We need a way to encode the data as qubits. This process is crucial if we want to obtain an effective quantum model. We usually refer to this mapping as data encoding, data embedding, or data loading and this is the role of the feature map. While feature mapping is a common ML mechanism, this process of loading data into quantum states does not appear in classical machine learning as that only operates in the classical world.
# 
# Once the data is loaded, we must immediately apply a parameterized quantum circuit. This circuit is a direct analog to the layers in classical neural networks. It has a set of tunable parameters or weights. The weights are optimized such that they minimize an objective function. This objective function characterizes the distance between the predictions and known labeled data. A parameterized quantum circuit is also called a parameterized trial state, variational form, or ansatz. Perhaps, the latter is the most widely used term.
# 
# For more information, we direct the reader to the [Quantum Machine Learning Course GitHub](https://github.com/Qiskit/textbook/tree/main/notebooks/quantum-machine-learning#).
# 
# Our choice of feature map will be the ZZ feature map. The ``zz_feature_map`` is one of the standard feature map circuits in the Qiskit circuit library. We pass `num_features` as `feature_dimension`, meaning the feature map will have `num_features` or `4` qubits.
# 
# We decompose the feature map into its constituent gates to give the reader a flavor of how feature maps may look.

# In[9]:


from qiskit.circuit.library import zz_feature_map

num_features = features.shape[1]

feature_map = zz_feature_map(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", style="clifford", fold=20)


# If you look closely at the feature map diagram, you will notice parameters `x[0], ..., x[3]`. These are placeholders for our features.
# 
# Now we create and plot our ansatz. Pay attention to the repetitive structure of the ansatz circuit. We define the number of these repetitions using the `reps` parameter.

# In[10]:


from qiskit.circuit.library import real_amplitudes

ansatz = real_amplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", style="clifford", fold=20)


# This circuit has 16 parameters named `θ[0], ..., θ[15]`. These are the trainable weights of the classifier.
# 
# We then choose an optimization algorithm to use in the training process. This step is similar to what you may find in classical deep learning frameworks. To make the training process faster, we choose a gradient-free optimizer. You may explore other optimizers available in Qiskit.

# In[11]:


from qiskit_machine_learning.optimizers import COBYLA

optimizer = COBYLA(maxiter=100)


# In the next step, we define where to train our classifier. We can train on a simulator or a real quantum computer. Here, we will use a simulator. We create an instance of the `Sampler` primitive from `StatevectorSampler` instance from `qiskit.primitives`. This is the reference implementation that is statevector based. Using qiskit runtime services you can create a sampler that is backed by a quantum computer.

# In[12]:


from qiskit.primitives import StatevectorSampler as Sampler

sampler = Sampler()


# We will add a callback function called `callback_graph`. `VQC` will call this function for each evaluation of the objective function with two parameters: the current weights and the value of the objective function at those weights. Our callback will append the value of the objective function to an array so we can plot the iteration versus the objective function value. The callback will update the plot at each iteration. Note that you can do whatever you want inside a callback function, so long as it has the two-parameter signature we mentioned above.

# In[13]:


from matplotlib import pyplot as plt
from IPython.display import clear_output

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()


# Now we are ready to construct the classifier and fit it. 
# 
# `VQC` stands for "variational quantum classifier." It takes a feature map and an ansatz and constructs a quantum neural network automatically. In the simplest case it is enough to pass the number of qubits and a quantum instance to construct a valid classifier. You may omit the `sampler` parameter, in this case a `Sampler` instance will be created for you in the way we created it earlier. We created it manually for illustrative purposes only.
# 
# Training may take some time. Please, be patient.

# In[14]:


import time
from qiskit_machine_learning.algorithms.classifiers import VQC

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")


# Let's see how the quantum model performs on the real-life dataset.

# In[15]:


train_score_q4 = vqc.score(train_features, train_labels)
test_score_q4 = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")


# As we can see, the scores are high, and the model can be used to predict labels on unseen data.
# 
# Now let's see what we can tune to get even better models.
# 
# - The key components are the feature map and the ansatz. You can tweak parameters. In our case, you may change the `reps` parameter that specifies how repetitions of a gate pattern we add to the circuit. Larger values lead to more entanglement operations and more parameters. Thus, the model can be more flexible, but the higher number of parameters also adds complexity, and training such a model usually takes more time. Furthermore, we may end up overfitting the model. You can try the other feature maps and ansatzes available in the [Qiskit circuit library](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit_library#n-local-circuits), or you can come up with custom circuits.
# - You may try other optimizers. Qiskit contains a bunch of them. Some of them are gradient-free, others not. If you choose a gradient-based optimizer, e.g., `L_BFGS_B`, expect the training time to increase. Additionally to the objective function, these optimizers must evaluate the gradient with respect to the training parameters, which leads to an increased number of circuit executions per iteration.
# - Another option is to randomly (or deterministically) sample `initial_point` and fit the model several times.
# 
# But what if a dataset contains more features than a modern quantum computer can handle? Recall, in this example, we had the same number of qubits as the number of features in the dataset, but this may not always be the case.
# 
# ## 4. Reducing the Number of Features
# 
# In this section, we reduce the number of features in our dataset and train our models again. We'll move through faster this time as the steps are the same except for the first, where we apply a PCA transformation. 
# 
# We transform our four features into two features only. This dimensionality reduction is for educational purposes only. As you saw in the previous section, we can train a quantum model using all four features from the dataset.
# 
# Now, we can easily plot these two features on a single figure.

# In[16]:


from sklearn.decomposition import PCA

features = PCA(n_components=2).fit_transform(features)

plt.rcParams["figure.figsize"] = (6, 6)
sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, palette="tab10")


# As usual, we split the dataset first, then fit a classical model.

# In[17]:


train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=algorithm_globals.random_seed
)

svc.fit(train_features, train_labels)

train_score_c2 = svc.score(train_features, train_labels)
test_score_c2 = svc.score(test_features, test_labels)

print(f"Classical SVC on the training dataset: {train_score_c2:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c2:.2f}")


# The results are still good but slightly worse compared to the initial version. Let's see how a quantum model deals with them. As we now have two qubits, we must recreate the feature map and ansatz.

# In[18]:


num_features = features.shape[1]

feature_map = zz_feature_map(feature_dimension=num_features, reps=1)
ansatz = real_amplitudes(num_qubits=num_features, reps=3)


# We also reduce the maximum number of iterations we run the optimization process for, as we expect it to converge faster because we now have fewer qubits.

# In[19]:


optimizer = COBYLA(maxiter=40)


# Now we construct a quantum classifier from the new parameters and train it. 

# In[20]:


vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []

# make the objective function plot look nicer.
plt.rcParams["figure.figsize"] = (12, 6)


start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")


# In[21]:


train_score_q2_ra = vqc.score(train_features, train_labels)
test_score_q2_ra = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset using real_amplitudes: {train_score_q2_ra:.2f}")
print(f"Quantum VQC on the test dataset using real_amplitudes:     {test_score_q2_ra:.2f}")


# Well, the scores are higher than a fair coin toss but could be better. The objective function is almost flat towards the end, meaning increasing the number of iterations won't help, and model performance will stay the same. Let's see what we can do with another ansatz.

# In[22]:


from qiskit.circuit.library import efficient_su2

ansatz = efficient_su2(num_qubits=num_features, reps=3)
optimizer = COBYLA(maxiter=40)

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")


# In[23]:


train_score_q2_eff = vqc.score(train_features, train_labels)
test_score_q2_eff = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset using efficient_su2: {train_score_q2_eff:.2f}")
print(f"Quantum VQC on the test dataset using efficient_su2:     {test_score_q2_eff:.2f}")


# The scores are better than in the previous setup. Perhaps if we had used more iterations, we could do even better.

# ## 5. Conclusion
# 
# In this tutorial, we have built two classical and three quantum machine learning models. Let's print an overall table with our results.

# In[24]:


print(f"Model                            | Test Score | Train Score")
print(f"SVC, 4 features                  | {train_score_c4:10.2f} | {test_score_c4:10.2f}")
print(f"VQC, 4 features, real_amplitudes | {train_score_q4:10.2f} | {test_score_q4:10.2f}")
print(f"-----------------------------------------------------------")
print(f"SVC, 2 features                  | {train_score_c2:10.2f} | {test_score_c2:10.2f}")
print(f"VQC, 2 features, real_amplitudes | {train_score_q2_ra:10.2f} | {test_score_q2_ra:10.2f}")
print(f"VQC, 2 features, efficient_su2   | {train_score_q2_eff:10.2f} | {test_score_q2_eff:10.2f}")


# Unsurprisingly, the classical models perform better than their quantum counterparts, but classical ML has come a long way, and quantum ML has yet to reach that level of maturity. As we can see,  we achieved the best results using a classical support vector machine. But the quantum model trained on four features was also quite good. When we reduced the number of features, the performance of all models went down as expected. So, if resources permit training a model on a full-featured dataset without any reduction, you should train such a model. If not, you may expect to compromise between dataset size, training time, and score.
# 
# Another observation is that even a simple ansatz change can lead to better results. The two-feature model with the `efficient_su2` ansatz performs better than the one with `real_amplitudes`. That means the choice of hyperparameters plays the same critical role in quantum ML as in classical ML, and searching for optimal hyperparameters may take a long time. You may apply the same techniques we use in classical ML, such as random/grid or more sophisticated approaches.
# 
# We hope this brief tutorial helps you to take the leap from classical to quantum ML.

# In[25]:


import tutorial_magics

get_ipython().run_line_magic('qiskit_version_table', '')
get_ipython().run_line_magic('qiskit_copyright', '')


# In[ ]:




