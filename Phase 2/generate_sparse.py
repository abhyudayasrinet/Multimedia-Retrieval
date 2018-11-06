import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
import os

# Following script converts the textual descriptor files to sparse matrices for users/images/locations and stores them in the respective fodlers

user_data = r"./desctxt/devset_textTermsPerUser.txt"
image_data = r"./desctxt/devset_textTermsPerImage.txt"
location_data = r"./desctxt/devset_textTermsPerPOI.txt" 

data_files = [user_data, image_data, location_data]
terms = set()
for data_file in data_files:
    filename = data_file
    data_file = open(filename, 'r', encoding='utf-8')
    
    for line in data_file.readlines():
        if(filename == location_data):
            quote_index = line.index("\"") 
            elements = line[quote_index:].strip().split(' ') #skip the name of the location
        else:
            elements = line.strip().split(' ')[1:] #skip the first element (id)
        for i in range(0,len(elements),4): #iterate in steps of 4 to get terms
            terms.add(elements[i][1:-1]) #remove quotes("xyz")

terms = list(terms)
terms.sort()

user_tfidf_data = []
user_row_indices = []
user_col_indices = []
user_row_dict = {}
user_col_dict = {}
user_id_dict = {}

image_tfidf_data = []
image_row_indices = []
image_col_indices = []
image_row_dict = {}
image_col_dict = {}
image_id_dict = {}

location_tfidf_data = []
location_row_indices = []
location_col_indices = []
location_row_dict = {}
location_col_dict = {}
location_id_dict = {}

for data_file in data_files:
    filename = data_file
    data_file = open(filename, 'r', encoding='utf-8')
    
    #arrays to store tf,df,tfidf values
    tfidf_data = []

    #array to coordinate indices for sparse matrix
    row_indices = []
    col_indices = []
    row_index = 0
    #dictionaries to store mapping of value=>numeric_index of sparse matrix
    row_dict = {} #id->numeric_index
    id_dict = {} #numeric_index->id
    col_dict = {term:index for index,term in enumerate(terms)} #term->numeric_index

    #populating the sparse matrix
    data_file.seek(0)
    for line in data_file.readlines():
        if(filename == location_data):
            quote_index = line.index("\"")
            index_name = line[:quote_index].strip()
            elements = line[quote_index:].strip().split(' ')
        else:
            elements = line.strip().split(' ')
            index_name = elements[0]
            elements = elements[1:]

        row_dict[index_name] = row_index
        id_dict[row_index] = index_name
        for i in range(0, len(elements), 4):
            #extract each term and their tf,df,tfidf values
            term = elements[i][1:-1] 
            tfidf = float(elements[i+3])
            #add term and indices to respective arrays
            col_index = col_dict[term]
            tfidf_data.append(tfidf)
            row_indices.append(row_index)
            col_indices.append(col_index)
        row_index += 1

    #save the created matrices and index mapping to the respective global variables
    if filename == user_data:
        user_tfidf_data = tfidf_data
        user_row_indices = row_indices
        user_col_indices = col_indices
        user_row_dict = row_dict
        user_col_dict = col_dict
        user_id_dict = id_dict
    elif filename == image_data:
        image_tfidf_data = tfidf_data
        image_row_indices = row_indices
        image_col_indices = col_indices
        image_row_dict = row_dict
        image_col_dict = col_dict
        image_id_dict = id_dict
    elif filename == location_data:
        location_tfidf_data = tfidf_data
        location_row_indices = row_indices
        location_col_indices = col_indices
        location_row_dict = row_dict
        location_col_dict = col_dict
        location_id_dict = id_dict

#get maximum indexed term used in the vector space
user_max = max(user_col_indices)
image_max = max(image_col_indices)
location_max = max(location_col_indices)
global_max = max([user_max, image_max, location_max])

#if a vector space matrix does not have the maximum indexed term then add that term with extremely small tfidf value to make the matrix dimensions common across all vector spaces

#pad the user matrix with small values 
if(user_max < global_max):
    for i in range(user_max, global_max+1):
        user_tfidf_data.append(0.00000000000001)
        user_row_indices.append(0)
        user_col_indices.append(i)

#pad the image matrix with small values
if(image_max < global_max):
    for i in range(image_max, global_max+1):
        image_tfidf_data.append(0.00000000000001)
        image_row_indices.append(0)
        image_col_indices.append(i)

#pad the location matrix with small values
if(location_max < global_max):
    for i in range(location_max, global_max+1):
        location_tfidf_data.append(0.00000000000001)
        location_row_indices.append(0)
        location_col_indices.append(i)

for data_file in data_files:    
    #create and save the sparse matrix and the mapping dictionaries for the user vector space
    if data_file == user_data:
        output_folder = "users"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        user_tfidf = sparse.csr_matrix((user_tfidf_data, (user_row_indices, user_col_indices)))
        np.save(output_folder+'/row_dict.npy', user_row_dict)
        np.save(output_folder+'/col_dict.npy', user_col_dict)
        np.save(output_folder+'/id_dict.npy', user_id_dict)
        sparse.save_npz(output_folder+"/tfidf_sparse.npz", user_tfidf)
    #create and save the sparse matrix and the mapping dictionaries for the image vector space
    elif data_file == image_data:
        output_folder = "images"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        image_tfidf = sparse.csr_matrix((image_tfidf_data, (image_row_indices, image_col_indices)))
        np.save(output_folder+'/row_dict.npy', image_row_dict)
        np.save(output_folder+'/col_dict.npy', image_col_dict)
        np.save(output_folder+'/id_dict.npy', image_id_dict)
        sparse.save_npz(output_folder+"/tfidf_sparse.npz", image_tfidf)
    #create and save the sparse matrix and the mapping dictionaries for the location vector space
    elif data_file == location_data:
        output_folder = "locations"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        location_tfidf = sparse.csr_matrix((location_tfidf_data, (location_row_indices, location_col_indices)))   
        np.save(output_folder+'/row_dict.npy', location_row_dict)
        np.save(output_folder+'/col_dict.npy', location_col_dict)
        np.save(output_folder+'/id_dict.npy', location_id_dict)
        sparse.save_npz(output_folder+"/tfidf_sparse.npz", location_tfidf)