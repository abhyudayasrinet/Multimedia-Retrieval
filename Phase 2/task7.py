import numpy as np
from scipy import sparse as sp
from tensorly.decomposition import parafac
from sklearn.cluster import KMeans
import timeit

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

if __name__ == "__main__":
    start = timeit.default_timer()

    load_data()

    rank = int(input("Enter k: "))

    #get list of term indices that exist for each user
    user_term_columns = []
    for i in range(user_tfidf.shape[0]):
        term_list = set(user_tfidf.getrow(i).nonzero()[1])
        user_term_columns.append(term_list)

    #get list of term indices that exist for each image
    image_term_columns = []
    for j in range(image_tfidf.shape[0]):
        term_list = set(image_tfidf.getrow(j).nonzero()[1])
        image_term_columns.append(term_list)

    #get list of term indices that exist for each location
    location_term_columns = []
    for k in range(location_tfidf.shape[0]):
        term_list = set(location_tfidf.getrow(k).nonzero()[1])
        location_term_columns.append(term_list)

    # #Create the user x image x location tensor
    tensor = np.zeros((user_tfidf.shape[0], image_tfidf.shape[0], location_tfidf.shape[0]))
    for i in range(user_tfidf.shape[0]):
        for j in range(image_tfidf.shape[0]):
            for k in range(location_tfidf.shape[0]):
                tensor[i,j,k] = len(user_term_columns[i].intersection(image_term_columns[j], location_term_columns[k]))
    
    users, images, locations = parafac(tensor, rank=rank) #perform cp decomposition
    
    f = open("task7_groups.txt","w")
    f.write("USERS\n")
    kmeans = KMeans(n_clusters=rank, random_state=0) #perform k means to get k groups of users
    kmeans.fit(users)
    user_labels = kmeans.labels_
    for i in range(rank):
        f.write("GROUP "+str(i+1)+"\n")
        indices = np.argwhere(user_labels == i).reshape(1,-1)
        for index in indices[0]:
            f.write(user_id_dict[index]+",")
        f.write('\n')
    
    f.write("IMAGES\n")
    kmeans = KMeans(n_clusters=rank, random_state=0) #perform k means to get k groups of images
    kmeans.fit(images)
    image_labels = kmeans.labels_
    for i in range(rank):
        f.write("GROUP "+str(i+1)+"\n")
        indices = np.argwhere(image_labels == i).reshape(1,-1)
        for index in indices[0]:
            f.write(image_id_dict[index]+",")
        f.write('\n')
    
    f.write("LOCATIONS\n")
    kmeans = KMeans(n_clusters=rank, random_state=0) #perform k means to get k groups of locations
    kmeans.fit(locations)
    location_labels = kmeans.labels_
    for i in range(rank):
        f.write("GROUP "+str(i+1)+"\n")
        indices = np.argwhere(location_labels == i).reshape(1,-1)
        for index in indices[0]:
            f.write(location_id_dict[index]+",")
        f.write('\n')
    f.close()