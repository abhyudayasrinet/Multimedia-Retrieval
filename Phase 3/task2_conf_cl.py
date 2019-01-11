# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:57:30 2018

@author: lavee
"""

import pandas as pd
import csv
import sys
import time
import datetime
import random
from scipy import sparse    
from tqdm import tqdm
from common import store_results


"""
The confidence clustering algorithm take the following user input:
    1. c - The number of clusters to identify
    
The image-image similarity graph created as an output of task1 is partitioned into 'c' clusters
"""    
        
#Start here     
start_time=time.time()

#Capture command line inputs
input_c = input("Enter number of clusters\n")
input_c = int(input_c)

#create a dictionary of the images taken as input with key as the index and value as the image_id
dict_temp = pd.read_csv('IndexToImage.csv', header=0, index_col=0, squeeze=True,encoding = "utf-8").to_dict()
dict_image_id = dict((str(k),str(v)) for k,v in dict_temp.items()) 


#Create an inverted dictionary with key as image_id and value as index
inv_dict_image_id = {v: k for k, v in dict_image_id.items()}

#Read the output of task1
filepath = r"./ImageSimilarities.csv"

image_similarity = pd.read_csv(filepath,sep = ",", header = None ,engine='python',encoding = "utf-8")
image_similarity=image_similarity.ix[1:,1:]

#create a dictionary to store the cluster and confidence values for each node
#The structure of dictionary is as followe=s:
#   Key: Image_ID
#   Value: List with first element as the cluster index and second value as teh confidence of belonging to that cluster

conf= {}

#convert the pandas dataframe to a numpy ndarray
image_sim_m = image_similarity.values

#choose a maximum value of the confidence
max_conf = 10

#iterate over all the images and randomly assign them to a cluster wuth some confidence 
for idx,key in enumerate(dict_image_id.keys()):
    col = random.randint(0,input_c-1)
    #c = random.randint(1, max_conf)
    c=max_conf//2
    conf[dict_image_id[key]] = [col,c]
    


#create a sparse matrix from the image similarities graph  
cx = sparse.coo_matrix(image_sim_m)

#iterate over all teh edges of the graph and decide accordingly change the cluster assignment and confidence values of the nodes
#The confidence clustering algorithm works as follows:
#We check whether for each edge, both the nodes belong to the same cluster. If they do, the confidence for both the nodes is incremented by 1
#If they do not belong to the same cluster, we check their confidence values.
#If both have positive confidencem it is decremented by one
#If one of them has zero confidence, it is moved to other node's cluster wiith zero confidence and the cocnfidence of the other node is incremented by 1
#If both of them have zero confidence, we randomly move one of the nodes to other's cluster with zero confidence and increment the other node's confidence by 1

for i,j,v in tqdm(zip(cx.row, cx.col, cx.data)):
    
    if(conf[dict_image_id[str(i)]][0] == conf[dict_image_id[str(j)]][0]):
        #print("already the same cluster")
        conf[dict_image_id[str(i)]][1] = min(max_conf,conf[dict_image_id[str(i)]][1] +1)
        conf[dict_image_id[str(j)]][1] = min(max_conf,conf[dict_image_id[str(j)]][1] +1)

    if(conf[dict_image_id[str(i)]][0] != conf[dict_image_id[str(j)]][0]):
        if (conf[dict_image_id[str(i)]][1] > 0  and  conf[dict_image_id[str(j)]][1] > 0):
            #print("decrementing confidence")
            
            conf[dict_image_id[str(i)]][1] = max(0,conf[dict_image_id[str(i)]][1] - 1)
            
            conf[dict_image_id[str(j)]][1] = max(0,conf[dict_image_id[str(j)]][1] - 1)
            
        elif(conf[dict_image_id[str(i)]][1] ==0  and  conf[dict_image_id[str(j)]][1] > 0):
            
            #print("changing clusters1")
            conf[dict_image_id[str(i)]][0] = conf[dict_image_id[str(j)]][0]
            
            conf[dict_image_id[str(j)]][1] = min(max_conf,conf[dict_image_id[str(j)]][1] + 1)
            
        elif(conf[dict_image_id[str(i)]][1] > 0  and  conf[dict_image_id[str(j)]][1] == 0):
            #print("changing clusters2")
            
            conf[dict_image_id[str(j)]][0] = conf[dict_image_id[str(i)]][0]
            conf[dict_image_id[str(i)]][1] = min(max_conf,conf[dict_image_id[str(i)]][1] + 1)
            
        elif(conf[dict_image_id[str(i)]][1] == 0  and  conf[dict_image_id[str(i)]][1] == 0):
        
            #print("changing clusters3")
            
            conf[dict_image_id[str(j)]][0] = conf[dict_image_id[str(i)]][0]
            conf[dict_image_id[str(i)]][1] = min(max_conf,conf[dict_image_id[str(i)]][1] + 1)
            
clusters = []
l_sort = []
dict_cluster_names = {}

for i in range(input_c):
    clusters.append([])
    l_sort.append([])
    dict_cluster_names[str(i)] = "Cluster" + str(i)
    
#add images to the repsective clusters
for  key, values in conf.items():
    l_sort[values[0]].append(int(inv_dict_image_id[key]))
    clusters[values[0]].append(key)
    
for x in range(len(clusters)):
    clusters[x] = [y for _,y in sorted(zip(l_sort[x],clusters[x]))]

clusters.sort(key=len)
#write output to a file
with open("task2_conf.csv",'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(clusters)
    
for i in range(input_c):
    print(len(clusters[i]))

    
store_results(clusters, "2.1", dict_cluster_names, False, "Confidence Clustering")