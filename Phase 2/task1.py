import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from scipy import sparse as sp

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
    global user_tfidf, user_row_dict, user_col_dict, user_id_dict, image_tfidf, image_row_dict, image_col_dict, image_id_dict,  location_tfidf, location_row_dict, location_col_dict, location_id_dict, location_name_dict, location_internal_name_dict

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

def task1(vector_space, data, col_dict, red_method, k):
    
    col_term_dict = {value:key for key,value in col_dict.items()} #reverse dictionary to get index->term mapping eg. 0 -> "abc"
    f = open("task1_" + vector_space + "_" + red_method +"_latentsemantics.csv", "w", encoding="utf-8") #output file
    
    if(red_method == "pca"): #perform PCA
        transformation = PCA(n_components = k)
        data = data.todense()
        principalComponents = transformation.fit(data)
        eigenvectors = principalComponents.components_
        #write term weights to file
        for i, eigenvector in enumerate(eigenvectors): 
            sorted_indices = np.argsort(eigenvector)[::-1]
            f.write("LATENT SEMANTIC " + str(i+1) + "\n")
            for j in sorted_indices:
                f.write(col_term_dict[j]+",")
            f.write('\n')
            for j in sorted_indices:
                f.write(str(eigenvector[j])+",")
            f.write('\n')
    
    elif(red_method == "svd"):#perform svd
        transformation = TruncatedSVD(n_components = k)
        principalComponents = transformation.fit(data)
        eigenvectors = principalComponents.components_
        #write term weights to file
        for i, eigenvector in enumerate(eigenvectors):
            f.write("LATENT SEMANTIC " + str(i+1) + "\n")
            sorted_indices = np.argsort(eigenvector)[::-1]
            for j in sorted_indices:
                f.write(col_term_dict[j]+",")
            f.write('\n')
            for j in sorted_indices:
                f.write(str(eigenvector[j])+",")
            f.write('\n')

    elif(red_method == "lda"):#perform LDA
        transformation = LatentDirichletAllocation(n_components=k)
        principalComponents = transformation.fit(data)
        lda_components = principalComponents.components_
        #write term weights to file
        for i, topic_vector in enumerate(lda_components):
            f.write("LATENT SEMANTIC " + str(i+1) + "\n")
            sorted_indices = np.argsort(topic_vector)[::-1]
            for j in sorted_indices:
                f.write(col_term_dict[j]+",")
            f.write('\n')
            for j in sorted_indices:
                f.write(str(topic_vector[j])+",")
            f.write('\n')
    f.close()

def main():

    vector_space = int(input("1. users 2. images 3. locations\n"))
    if(vector_space == 1):
        vector_space = "user"
        data = user_tfidf
        col_dict = user_col_dict
    elif(vector_space == 2):
        vector_space = "image"
        data = image_tfidf
        col_dict = image_col_dict
    else:
        vector_space = "location"
        data = location_tfidf
        col_dict = location_col_dict

    print(data)

    red_method = int(input("1. PCA 2. SVD 3.LDA\n"))
    if(red_method == 1):
        red_method = "pca"
    elif(red_method == 2):
        red_method = "svd"
    elif(red_method == 3):
        red_method = "lda"

    k = int(input("enter k: "))
    task1(vector_space, data, col_dict, red_method, k)

if __name__ == "__main__":
    load_data()
    main()     