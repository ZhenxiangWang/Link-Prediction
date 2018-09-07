# Link-Prediction
Link prediction is to predict the existence of a link between two nodes in a network.

## Dataset
The training network is a partial crawl of the Twitter social network from several years ago. The nodes
in the network—Twitter users—have been given randomly assigned IDs, and a directed edge from node A to
B represents that user A follows B. The training network is a subgraph of the entire network. Starting from
several random seed nodes, data maker proceeded to obtain the friends of the seeds, then their friends’ friends, and so
on for several iterations.

The test data is a list of 2,000 edges, and the task is to predict if each of those test edges are really edges in
the Twitter network or are fake ones. 1,000 of these test edges are real and withheld from the training network,
while the other 1,000 do not actually exist

## How to run?
1. Random Sampling

    ```python sampling.py```
2. Generate Directed Features

    ```python directed_features.py```
3. Generate Undirected Features

    ```python undirected_features.py```
4. Generate Experimental Features

    ```python experimental_features.py```
5. Logistic Regression

    ```run LogisticRegression.ipynb```
5. LightGBM

    ```run LightGBM.ipynb```
5. Deep Learning

    ```run DeepLearning.ipynb```
