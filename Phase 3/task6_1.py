import numpy as np
import pandas as pd
from collections import Counter
import csv
import random
from common import store_results

image_similarites_file='ImageSimilarities.csv'
idxToImage_file='IndexToImage.csv'
df=pd.read_csv(image_similarites_file,index_col=None)
idxToImage=pd.read_csv(idxToImage_file,index_col=0)
idxToImage = idxToImage.values.tolist()

df=df.values[:,1:]

#Get imags with labels and names
def getUniqueLabels(train_images):
    unique_labels=[]
    for i in range(len(train_images)):
        if train_images[i][1] not in unique_labels:
            unique_labels.append(train_images[i][1])
    return unique_labels

#get list of images 
def getImages(filename):
    f=open(filename)
    rows=f.readlines()
    names=[]
    images=[]
    for row in rows:
        row=row.strip()
        image=row.split(" ")[0]
        label=row.split(" ")[1]
        images.append([image,label])
        names.append(image)
    return images,names
    
def getLabelsToIdx(train_labels):
    train_labels=np.asarray(train_labels)
    unique_labels=np.unique(train_labels)
    labelsToIdx={}
    for i in range(len(unique_labels)):
        labelsToIdx[unique_labels[i]]=i
    return labelsToIdx

def getIdxToLabels(labelsToIdx):
    idxToLabels={}
    i=0
    for keys in labelsToIdx:
        idxToLabels[str(labelsToIdx[keys])]=keys
        i=i+1
    return idxToLabels
    
def bfs(df,k,max_height,test_image,idxToImage,train_names,train_images,test_images):
    x=int(idxToImage.index(test_image))
    number_of_neighbours = len(np.where(df[x]>0)[0])
    immediate_neighbours = list(np.argsort(df[x]))
    immediate_neighbours.reverse()
    bfs_q = []
    visited = []
    for node in immediate_neighbours[:number_of_neighbours]:
        bfs_q.append((node, 0))
 
    while(len(bfs_q) > 0):

        curr_node = bfs_q[0][0]

        neighbour=str(idxToImage[curr_node])
        if neighbour in train_names:
            idx = train_names.index(neighbour)
            return train_images[idx][1]

        curr_height = bfs_q[0][1]
        if(curr_height >= max_height):
            break
        bfs_q = bfs_q[1:]
        if(curr_node in visited):
            continue
        visited.append(curr_node)
        number_of_curr_neighbours = len(np.where(df[curr_node] > 0)[0])
        curr_neighbours = list(np.argsort(df[curr_node]))
        curr_neighbours.reverse()
        for node in curr_neighbours[:number_of_curr_neighbours]:
            if(node not in visited):
                bfs_q.append((node, curr_height+1))
    
    unique_labels = getUniqueLabels(train_images)
    index = random.randint(0,len(unique_labels)-1)
    return unique_labels[index]

def getKneighbours(test_image, df, k, idxToImage):
    x=int(idxToImage.index(test_image))
    number_of_neighbours = len(np.where(df[x]>0)[0])
    immediate_neighbours = list(np.argsort(df[x]))
    immediate_neighbours.reverse()
    bfs_q = immediate_neighbours[:number_of_neighbours]
    kneighbours = []
    while(len(bfs_q) > 0 and len(kneighbours) != k):
        curr_node = bfs_q[0]
        bfs_q = bfs_q[1:]
        if(curr_node in kneighbours):
            continue
        kneighbours.append(curr_node)
        number_of_curr_neighbours = len(np.where(df[curr_node] > 0)[0])
        curr_neighbours = list(np.argsort(df[curr_node]))
        curr_neighbours.reverse()
        bfs_q.extend(curr_neighbours[:number_of_curr_neighbours])
    return kneighbours


input_file='task6-input.txt'
train_images,train_names=getImages(input_file)
unique_labels=getUniqueLabels(train_images)
labelsToIdx=getLabelsToIdx(unique_labels)
idxToLabels=getIdxToLabels(labelsToIdx)

test_images=[]
test_names=[]
for i in range(df.shape[0]):
    if str(idxToImage[i]) not in train_names:
        test_images.append([idxToImage[i],'-1'])
        test_names.append(idxToImage[i])

k=int(input("Enter k "))
for i,test_image in enumerate(test_names):
    similarity_score=[]
    immediate_neighbours=getKneighbours(test_image,df,k,idxToImage)
    
    labels=[]
    for j in range(len(immediate_neighbours)):
        max_height=2
        y=immediate_neighbours[j]
        neighbour=str(idxToImage[y])
        if neighbour in train_names:
            labels.append(train_images[y][1])
        else:
            labels.append(bfs(df,k,max_height,test_image,idxToImage,train_names,train_images,test_images))
            
    count=Counter(labels)
    test_images[i][1]=count.most_common(1)[0][0]

data=[0]*len(unique_labels)

for label in unique_labels:
    temp=[] 
    for i in range(len(train_images)):
        if train_images[i][1] == label:
            temp.append(train_images[i][0])
    for i in range(len(test_images)):
        if test_images[i][1] == label:
            temp.append(test_images[i][0])
    data[labelsToIdx[label]]=temp

    
f=open('task6-knn-output.txt','w')
for i in range(len(unique_labels)):
    f.write(str(idxToLabels[str(i)]))
    f.write('\n')
    f.write(str(data[i]))
    f.write('\n')
    f.write('\n')
f.close()

store_results(data, "6.1", idxToLabels, False, "KNN")