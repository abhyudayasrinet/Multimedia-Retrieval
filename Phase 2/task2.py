import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
import sklearn
from scipy import sparse as sp
import pandas as pd

user_tfidf = None #tfidf values per user
user_id_dict = None #numeric_index->id eg. 0->10117222@N04
user_col_dict = None #term->numeric_index eg. argentina->3
user_row_dict = None #id->numeric_index eg. eg. 10117222@N04->0
image_tfidf = None #tfidf values per image
image_id_dict = None #numeric_index->id eg. 0->9067739127
image_col_dict = None #term->numeric_index eg. argentina->3
image_row_dict = None #id->numeric_index eg. eg. 9067739127->0
location_tfidf = None #tfidf values per location
location_id_dict = None #numeric_index->id eg. 0->acropolis athens
location_col_dict = None #term->numeric_index eg. argentina->3
location_row_dict = None #id->numeric_index eg. eg. acropolis athens->0
location_name_dict = None #location_id->location_name(text) eg. 1->angel of the north
location_internal_name_dict = None #location_id->location_internal_name eg. 1->angel_of_the_north

def load_data():
    """
    Loads the preprocessed sparse matrices and dictionary files to respective global variables
    """
    global user_tfidf, user_row_dict, user_col_dict, user_id_dict, image_tfidf, image_row_dict, image_col_dict, image_id_dict,  location_tfidf,location_row_dict, location_col_dict, location_id_dict, location_name_dict, location_internal_name_dict

    user_tfidf = sp.load_npz("users/tfidf_sparse.npz")
    user_id_dict = np.load("users/id_dict.npy").item()
    user_col_dict = np.load("users/col_dict.npy").item()
    user_row_dict = np.load("users/row_dict.npy").item()
    image_tfidf = sp.load_npz("images/tfidf_sparse.npz")
    image_id_dict = np.load("images/id_dict.npy").item()
    image_col_dict = np.load("images/col_dict.npy").item()
    image_row_dict = np.load("images/row_dict.npy").item()
    location_tfidf = sp.load_npz("locations/tfidf_sparse.npz")
    location_id_dict = np.load("locations/id_dict.npy").item()
    location_col_dict = np.load("locations/col_dict.npy").item()
    location_row_dict = np.load("locations/row_dict.npy").item()
    location_name_dict = np.load("locations/location_name_dict.npy").item()
    location_internal_name_dict = np.load("locations/location_internal_name_dict.npy").item()

def task2(transformation):
    """
    Projects the user,image and location vector space onto the new latent semantics and uses cosine similarity to identify the the top 5 closest users, images and locations
    """
    global image_tfidf, user_tfidf, location_tfidf, image_id_dict, location_id_dict, user_id_dict, red_method, vector_space, image_row_dict, user_row_dict, location_row_dict, location_name_dict

    #project each vector space to new semantics
    if(red_method == "pca"):
        reduced_user_data = transformation.transform(user_tfidf.todense())
        reduced_image_data = transformation.transform(image_tfidf.todense())
        reduced_location_data = transformation.transform(location_tfidf.todense())
    elif(red_method == "svd" or red_method == "lda"):
        reduced_user_data = transformation.transform(user_tfidf)
        reduced_image_data = transformation.transform(image_tfidf)
        reduced_location_data = transformation.transform(location_tfidf)

    for input_type in ["user", "image", "location"]:
        
        if(input_type == "user"):
            #get input user id
            input_id = input("Enter user id: ")
            input_index = user_row_dict[input_id]
            input_vector = reduced_user_data[input_index,:]
        elif(input_type == "image"):
            #get input image id
            input_id = input("Enter image id: ")
            input_index = image_row_dict[input_id]
            input_vector = reduced_image_data[input_index,:]
        elif(input_type == "location"):
            #get input location id
            input_id = int(input("Enter location id: "))
            location_name = location_name_dict[input_id]
            input_index = location_row_dict[location_name]
            input_vector = reduced_location_data[input_index,:]

        
        #get similar users
        similarities = sklearn.metrics.pairwise.cosine_similarity(reduced_user_data, input_vector.reshape(1, -1))
        distances = (1 - similarities)/2
        distances = np.absolute(distances)
        sorted_indices = np.argsort(distances[:,0])
        print("Top 5 similar users")
        for i in range(0,5):
            print(user_id_dict[sorted_indices[i]], distances[sorted_indices[i]])
        print()
        
        #get similar images
        similarities = sklearn.metrics.pairwise.cosine_similarity(reduced_image_data, input_vector.reshape(1, -1))
        distances = (1 - similarities)/2
        distances = np.absolute(distances)
        sorted_indices = np.argsort(distances[:,0])
        print("Top 5 similar images")
        for i in range(0,5):
            print(image_id_dict[sorted_indices[i]], distances[sorted_indices[i]])
        print()

        #get similar locations
        similarities = sklearn.metrics.pairwise.cosine_similarity(reduced_location_data, input_vector.reshape(1, -1))
        distances = (1 - similarities)/2
        distances = np.absolute(distances)
        sorted_indices = np.argsort(distances[:,0])
        print("Top 5 similar locations")
        for i in range(0,5):
            print(location_id_dict[sorted_indices[i]], distances[sorted_indices[i]])
        print()

def task1(data):
    """
    Performs the chosen reduction method (PCA, SVD, LDA) on the chosen vector space to identify the latent semantics
    """
    global red_method, k, vector_space

    if(red_method == "pca"): #perform pca
        transformation = PCA(n_components = k)
        data = data.todense()
        transformation.fit(data)
    
    elif(red_method == "svd"): #perform svd
        transformation = TruncatedSVD(n_components = k)
        transformation.fit(data)

    elif(red_method == "lda"): #perform lda
        transformation = LatentDirichletAllocation(n_components=k)
        transformation.fit(data)

    task2(transformation)


red_method = None #chosen reduction method (PCA, SVD, LDA)
k = None #number of latent semantics to find
vector_space = None #vector space to perform latent semantics analysis on (user, images, locations)

if __name__ == "__main__":
    load_data()

    vector_space = int(input("1. users 2. images 3. locations\n"))
    if(vector_space == 1):
        vector_space = "user"
        data = user_tfidf
    elif(vector_space == 2):
        vector_space = "image"
        data = image_tfidf
    else:
        vector_space = "location"
        data = location_tfidf

    red_method = int(input("1. PCA 2. SVD 3.LDA\n"))
    if(red_method == 1):
        red_method = "pca"
    elif(red_method == 2):
        red_method = "svd"
    elif(red_method == 3):
        red_method = "lda"

    k = int(input("enter k: "))
    task1(data)