# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 19:56:43 2018

@author: lavee
"""

import pandas as pd
import csv
import numpy as np
import sys
import time
import datetime
from numpy import linalg as LA
from common import store_results
from tqdm import tqdm


"""
The spectral clustering algorithm take the following user input:
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

#Replace the edge weights with 1
image_similarity[image_similarity != 0] = 1



#create the degree matrix
df_degree = list(image_similarity.astype(bool).sum(axis=1))
deg_arr = np.diag(df_degree)

image_sim_m= image_similarity.values

#create laplacian matrix
lap = np.subtract(deg_arr,image_sim_m)

#Eigen values and eigen vectors of laplacian matrix
eigenValues, eigenVectors = LA.eig(lap)

#sort the eigen values in descending order
idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx].real
eigenVectors = eigenVectors[:,idx].real

clusters=[]
e = eigenVectors[:,-2]


clusters.append([])
clusters.append([])

po = np.sum(np.array(e) > 0, axis=0)
ne = np.sum(np.array(e) < 0, axis=0)
ze = np.sum(np.array(e) == 0, axis=0)

#Assign nodes to one of the two clusters based on the corresponding values in eigen vector
for i in range(len(e)):
    c1=0
    c2=1
    if(ne==0):
        if(e[i]>0):
            clusters[c1].append(dict_image_id[str(i)])
        else:       
            clusters[c2].append(dict_image_id[str(i)])
    else:
        if(e[i]>=0):
            clusters[c1].append(dict_image_id[str(i)])
        else:       
            clusters[c2].append(dict_image_id[str(i)])  
c=2


eigen = []

for i in range(input_c-2):
        print("outer iteration ")
        min_ev = sys.maxsize
        cl = -1
        #compare the eigen values of all the clusters to decide which one to divide next
        for j in range(c):
                print("inner iteration ")
                #create image similarity matrix for the current cluster
                image_similarity_temp = image_similarity.ix[list(int(inv_dict_image_id[val]) for val in clusters[j]),list(int(inv_dict_image_id[val]) for val in clusters[j])]
                
                #create degree matrix for the current cluster
                df_degree_temp = list(image_similarity_temp.astype(bool).sum(axis=1))
                deg_arr_temp = np.diag(df_degree_temp)
                
                image_sim_temp = image_sim_m[list(int(inv_dict_image_id[val]) for val in clusters[j]),:][:,list(int(inv_dict_image_id[val]) for val in clusters[j])]
                
                #create laplacian matrix for the current cluster
                lap_temp = np.subtract(deg_arr_temp,image_sim_temp)
                
                #eigen decomposition
                eigenValues_temp, eigenVectors_temp = LA.eig(lap_temp)
                
                idx_temp = eigenValues_temp.argsort()[::-1]   
                eigenValues_temp = eigenValues_temp[idx_temp]
                eigenVectors_temp = eigenVectors_temp[:,idx_temp]
                if(eigenValues_temp.size>1):
                    po = np.sum(np.array(eigenVectors_temp[:,-2]) > 0, axis=0)
                    ne = np.sum(np.array(eigenVectors_temp[:,-2]) < 0, axis=0)
                    ze = np.sum(np.array(eigenVectors_temp[:,-2]) == 0, axis=0)
                    print(po)
                    print(ne)
                    print(ze)
                    if(po != 0 and ne==0 and ze==0):
                        continue
                    if(po == 0 and ne!=0 and ze==0):
                        continue
                    if(po == 0 and ne==0 and ze!=0):    
                        continue
                    

                if(eigenValues_temp.size>1 and eigenValues_temp[-2]<=min_ev ):
                    min_ev = eigenValues_temp[-2]
                    cl=j
                    min_eigenVectors_temp=eigenVectors_temp

        e_temp = min_eigenVectors_temp[:,-2]
        cluster_part = clusters[cl]
        
        po = np.sum(np.array(e_temp) > 0, axis=0)
        ne = np.sum(np.array(e_temp) < 0, axis=0)
        ze = np.sum(np.array(e_temp) == 0, axis=0)

        
        del clusters[cl]
       
        clusters.append([])
        clusters.append([])
        
        #Assign nodes to one of the two clusters based on the corresponding values in eigen vector
        for k in range(len(e_temp)):
            c1=c-1
            c2=c
            if(ne==0):
                if(e_temp[k]>0):
                    clusters[c1].append(cluster_part[k])
                else:
                    
                    clusters[c2].append(cluster_part[k])
            else:
                if(e_temp[k]>=0):
                    clusters[c1].append(cluster_part[k])
                else:
                    
                    clusters[c2].append(cluster_part[k])
        c += 1
       
l_sort=[]
dict_cluster_names = {}

for i in range(input_c):
    l_sort.append([])
    dict_cluster_names[str(i)] = "Cluster" + str(i)

    
#add images to the repsective clusters
for  idx,item in enumerate(clusters):
    for i in range(len(item)):
        l_sort[idx].append(int(inv_dict_image_id[item[i]]))
    
for x in range(len(clusters)):
    clusters[x] = [y for _,y in sorted(zip(l_sort[x],clusters[x]))]

#clusters.sort(key=len)
#print(clusters)
clusters.sort(key=len)    
with open("task2_spectral.csv",'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(clusters)
  
for i in range(input_c):
    print(len(clusters[i]))

store_results(clusters, "2.2", dict_cluster_names, False, "Spectral Clustering")