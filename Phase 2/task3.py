import xml.etree.ElementTree as ET
import csv
import os.path
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import euclidean_distances
from operator import sub

#This function returns list of image_ids, data (features) and location from csv file of visual descriptor
def getDictionary(filename, image_id, data,location_index,location):
	with open(filename) as csvfile:
		rows = csv.reader(csvfile)
		for row in rows:	
			image_id.append(row[0])
			data.append(row[1:])
			location_index.append(location)	
	return image_id, data,location_index

#Get Location name and location id from xml file
def parseLocationId():
	tree = ET.parse('./devset_topics.xml')
	root = tree.getroot()
	keys = []
	values = []
	for child in root:
		for grandchild in child:
			if(grandchild.tag == "number"):
				keys.append(grandchild.text)
			if(grandchild.tag == "title"):
				values.append(grandchild.text)
	return keys, values
#Given data reprenstation in latent semantics and imageID returns 5 most similar location and images
def task3(reduced_data, image_id,location_index,locations):
	input_id = input("Enter image ID: ")
	image_index = image_id.index(input_id)
	input_vector = reduced_data[image_index]
	#Find eucludean distance of each image with source image
	similarities = euclidean_distances(reduced_data, input_vector.reshape(1,-1))
	sorted_indices = np.argsort(similarities[:,0])
	for i in range(0, 5):
		print(image_id[sorted_indices[i]], similarities[sorted_indices[i]])
	location_input = location_index[image_id.index(input_id)]
	mean_scores=[]
	#Find similarity of each images in locations to source image and take average 
	for i in range(len(locations)):
		start_index = location_index.index(locations[i])
		end_index = start_index+location_index.count(locations[i])-1
		similarities = euclidean_distances(reduced_data[start_index:end_index+1], input_vector.reshape(1, -1))
		mean_scores.append(np.mean(similarities))

	sorted_indices = np.argsort(mean_scores)

	locationId, not_sorted_locations = parseLocationId()
	for i in range(0,5):
		locationId_index=not_sorted_locations.index(locations[sorted_indices[i]])
		print(locationId[locationId_index], locations[sorted_indices[i]], mean_scores[sorted_indices[i]])

#This function find k latent semantic method for given data matrix based on dimenonsality reduction method
def task1(data, image_id, red_method, k,location_index,locations):
	if(red_method == "pca"):
		transformation = PCA(n_components = k)
		for i in range(len(data)):
			for j in range(len(data[i])):
				data[i][j]=float(data[i][j])
		reduced_data = transformation.fit_transform(data)
		
	elif(red_method == "svd"):
		transformation = TruncatedSVD(n_components = k)
		for i in range(len(data)):
			for j in range(len(data[i])):
				data[i][j]=float(data[i][j])
		reduced_data = transformation.fit_transform(data)

	elif(red_method == "lda"):
		transformation = LatentDirichletAllocation(n_components=k)
		for i in range(len(data)):
			for j in range(len(data[i])):
				data[i][j]=float(data[i][j])
		reduced_data = transformation.fit_transform(data)

	task3(reduced_data, image_id,location_index,locations)

#Driver Function
def main():
	locationId, locations = parseLocationId()
	locations = sorted(locations)
	model = input("Enter model :")
	
	k = input("Enter k :")
	
	red_method = int(input("1. PCA 2. SVD 3.LDA\n"))
	
	if(red_method == 1):
		red_method = "pca"
	elif(red_method == 2):
		red_method = "svd"
	elif(red_method == 3):
		red_method = "lda"

	directory_path = "./descvis/img/"
	data = []
	row_indices = []
	column_indices = []
	image_id = []
	data = []
	location_index=[]
	for i in range(len(locations)):
		filename = directory_path + locations[i] + " " + str(model) + ".csv"
		image_id, data,location_index = getDictionary(filename, image_id, data,location_index,locations[i])
	
	#LDA don't accept negative values that's why we shifted all values by global optima to make all values positives
	if(red_method=="lda"):	
		data = np.array(data,dtype=float)
		
		global_min=0
		
		for idx,col in enumerate(data.T):
			global_min = (col).min()
			(data.T)[idx]=(data.T)[idx] - global_min
		data  = data.tolist()
	task1(data, image_id, red_method, int(k),location_index,locations)
	
if __name__ == "__main__":
	main()