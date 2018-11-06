# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:56:43 2018

@author: lavee
"""

import pandas as pd
import csv
from numpy.linalg import norm
import numpy as np
import sys
from bs4 import BeautifulSoup as Soup
from sklearn.metrics.pairwise import manhattan_distances
from numpy import genfromtxt
import sklearn
import itertools
import time
from sklearn.decomposition import TruncatedSVD
import datetime
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation


#Create a list of list containing location ids and location names by  processing the xml file
def location_preprocess():
	xml_file_name="devset_topics.xml"
	xml_file_location=r"./"

	visual_desc_location=r"./descvis/img/"

	datasource = open(xml_file_location + xml_file_name,encoding='utf-8').read()
	soup=Soup(datasource,'lxml')
	
	location=[]
	location_id=[]
	location_name=[]
	
	#listing location ids and location names
	for number in soup.findAll('number'):
		location_id.append(number.string)
	for title in soup.findAll('title'):
		location_name.append(title.string)
	
	location.append(location_id)
	location.append(location_name)
	
	return(location)

#Read file into a numpy 2D array
def read_file(filepath):
	return(genfromtxt(filepath, delimiter=',',encoding = "utf-8"))

	
#Finding distances between all image pairs of two locations	
def write_distances(input_file_data,other_location):
	
	#removing location id from the numpy array to use
	input_images=input_file_data[:,1]
	images_candidate=other_location[:,1]
	
	#location ids of input and the location it is being comoared to
	input_location=input_file_data[0,0]
	candidate_location=other_location[0,0]
	
	#Finding the pairwise n*m distance matrix 
	distance=pd.DataFrame()
	distance_pairwise = sklearn.metrics.pairwise.euclidean_distances(input_file_data[:,2:],other_location[:,2:])
	
	sz=np.size(distance_pairwise)
	
	product = list(itertools.product(input_images,images_candidate))
	a=np.array(product)
	sz_a=np.size(a)
		
	#reshaping the pairwise matrix to create (n*m)*2 matrix
	distance_pairwise=distance_pairwise.reshape(sz,1)
	distance = np.concatenate((distance_pairwise,a), axis=1)
	
	#creating final distance matrix by adding location names
	distance=np.insert(distance, 1, values=input_location, axis=1)
	distance=np.insert(distance, 3, values=candidate_location, axis=1)
	
	#sorting by distance
	distance_sorted = distance[distance[:,0].argsort()]
	return(pd.DataFrame(distance_sorted,columns=['Distance','Input Location','Input Id','Second Location','Second Id']))
	
#create a dictionary to create a mapping between location id and location name
def set_file_names(location,model):
	location_id=location[0]
	location_name=location[1]
	
	dict = {}
	
	for idx,key in enumerate(location_id):
			dict[key] = location_name[idx] + str(" ") + model + str(".csv")
			
	return(dict)
		

	
	
		
#Start here		
start_time=time.time()


#Capture command line inputs
input_id = input("Enter location Id\n")
input_model = input("Enter model\n")
input_k = input("Enter k\n")

location=location_preprocess()		
location_id = location[0]
location_name = location[1]

dm_algo=input("Input 1: PCA, 2: SVD, 3: LDA\n")


file_names = set_file_names(location,input_model)

target_file_location = r"./"
file_location = r"./descvis/img/"

#Read the input location file
input_file_name = file_names[input_id]
input_file_data = read_file(file_location + input_file_name)

#Insert Location ID as the first column
input_file_data = np.insert(input_file_data, 0, values=input_id, axis=1)

number_of_entities = len(file_names)

other_locations_data=[]

#first column contains the location id, second column has the image ids, so eliminate them before finding the latent features
input_file_features = input_file_data[:,2:]


#Reading other location files
idx = 0
for key in file_names:
		other_locations_data.append(read_file(file_location+file_names[key]))
		other_locations_data[idx]=np.insert(other_locations_data[idx],0, values=key,axis=1)
		idx = idx+1

#If performing LDA, shift the values by the global minima of each column to get values in positive range		
if(dm_algo == '3'):	
	for idx,col in enumerate(input_file_features.T):
		global_min=col.min()
		min_other = min((i.T)[idx+2].min() for i in other_locations_data)
		global_min=min(global_min,min_other)
		
		(input_file_features.T)[idx] = (input_file_features.T)[idx] - global_min
		for i in other_locations_data:
			(i.T)[idx+2] = (i.T)[idx+2] - global_min
		

if(dm_algo == '1'):
	#Apply PCA
	pca = PCA(n_components=int(input_k))
	val = pca.fit(input_file_features)
	new_data = pca.transform(input_file_features)

elif(dm_algo == '2'):
	#Apply SVD
	svd = TruncatedSVD(n_components = int(input_k))
	val = svd.fit(input_file_features)
	new_data = svd.transform(input_file_features)


elif(dm_algo == '3'):
	#Apply LDA
	lda = LatentDirichletAllocation(n_components=int(input_k), random_state=0)
	val = lda.fit(input_file_features)
	new_data = lda.transform(input_file_features)
	
#Input location in the transformed space
transformed_input_location = np.append(input_file_data[:,0:2],new_data,axis=1)

other_locations_transformed  = []


#Other locations in the transformed space
if(dm_algo == '1'):
	for data in other_locations_data:
		other_locations_transformed.append(np.append(data[:,0:2],pca.transform(data[:,2:]),axis=1))
elif(dm_algo == '2'):
	for data in other_locations_data:
		other_locations_transformed.append(np.append(data[:,0:2],svd.transform(data[:,2:]),axis=1))
elif(dm_algo == '3'):
	for data in other_locations_data:
		other_locations_transformed.append(np.append(data[:,0:2],lda.transform(data[:,2:]),axis=1))

#find distances of given location with other location in the transformed space
d=[]
for i in range(number_of_entities):
	d.append(write_distances(transformed_input_location,other_locations_transformed[i]))

df_distance=pd.DataFrame(columns=['input','candidate','avg_distance'])

#Find the mean of distances between all image pairs
for i in range(number_of_entities):
	pairwise_distance = d[i]
	avg_distance = pairwise_distance["Distance"].mean()
	input = pairwise_distance["Input Location"].iloc[0]
	candidate = pairwise_distance["Second Location"].iloc[0]
	input_pairs = []
	df_distance = df_distance.append([{'input':input,'candidate':candidate,'avg_distance':avg_distance}])

#Sort the distances based on avg distance, lesser the distance more the similarity
sorted_df = df_distance.sort_values('avg_distance',ascending=True)	

sorted_df = sorted_df.head(int(5))
output_df = pd.DataFrame()


#Replace the location id with location name
for idx,row in sorted_df.iterrows():
	row['candidate']=file_names[str(int(row['candidate']))].split(' ')[0]
	row['input']=input_file_name.split(' ')[0]
	output_df=output_df.append(row)

print(output_df)
sys.exit()
