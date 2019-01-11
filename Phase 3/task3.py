import numpy as np
import pandas as pd
from common import store_results

k = int(input("Enter k: "))

df = pd.read_csv("ImageSimilarities.csv", index_col=0) #Load output of task 1
id_map = pd.read_csv("IndexToImage.csv", index_col=0)
df = df.values #convert to numpy array
df[df.nonzero()] = 1 #convert similarity values to edge presence. 1 implying edge from node i to node j

nodes = df.shape[0]
for i in range(df.shape[0]): #handle dangling nodes
    if(df[i,:].sum() == 0):
        df[i,:] = 1/nodes

adj_matrix = df.T #take transpose so source is at column indices and destination is at row indices 
M = (adj_matrix / np.sum(adj_matrix, axis = 0)) #convert to column stochastic form
transportation_matrix = np.full((nodes, 1), 1/nodes) #transportation column vector of values 1/total_nodes

beta = 0.55 #damping factor
pagerank_scores = np.full((nodes, 1), 1/nodes) #initial pagerank score
epsilon = 0.000001

# calculate pagerank using power iteration method
while True:
        old_pagerank_scores = pagerank_scores
        pagerank_scores = (beta * np.dot(M, pagerank_scores)) + ((1 - beta) * transportation_matrix)
        if np.linalg.norm(pagerank_scores - old_pagerank_scores) < epsilon:
            break

#get top K dominant images based on pagerank scores
pagerank_scores = pagerank_scores.reshape(1, -1)
indices = np.argsort(pagerank_scores, axis = 1)
top_k_images = []
for i in range(1, k+1):
        top_k_images.append(str(id_map.iloc[indices[0][-i]][0]))

#report output to app
output = [top_k_images]
label_dict = { "0" : "K Dominant Images"}
print(output)
store_results(output, "3", label_dict, False, algo='PageRank')