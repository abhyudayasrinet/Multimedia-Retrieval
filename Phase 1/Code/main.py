import numpy as np
from scipy import sparse
import heapq
import pprint
import timeit
import multiprocessing

user_tf = None #tf values per user
user_df = None #df values per user
user_tfidf = None #tfidf values per user
user_id_dict = None #numeric_index->id eg. 0->10117222@N04
user_col_dict = None #term->numeric_index eg. argentina->3
user_row_dict = None #id->numeric_index eg. eg. 10117222@N04->0
image_tf = None #tf values per image
image_df = None #df values per image
image_tfidf = None #tfidf values per image
image_id_dict = None #numeric_index->id eg. 0->9067739127
image_col_dict = None #term->numeric_index eg. argentina->3
image_row_dict = None #id->numeric_index eg. eg. 9067739127->0
location_tf = None #tf values per location
location_df = None #df values per location
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
    global user_tf, user_df, user_tfidf, user_row_dict, user_col_dict, user_id_dict,image_tf, image_df, image_tfidf, image_row_dict, image_col_dict, image_id_dict, location_tf, location_df, location_tfidf,location_row_dict, location_col_dict, location_id_dict, location_name_dict, location_internal_name_dict

    user_tf = sparse.load_npz("users/tf_sparse.npz")
    user_df = sparse.load_npz("users/df_sparse.npz")
    user_tfidf = sparse.load_npz("users/tfidf_sparse.npz")
    user_id_dict = np.load("users/id_dict.npy").item()
    user_col_dict = np.load("users/col_dict.npy").item()
    user_row_dict = np.load("users/row_dict.npy").item()
    image_tf = sparse.load_npz("images/tf_sparse.npz")
    image_df = sparse.load_npz("images/df_sparse.npz")
    image_tfidf = sparse.load_npz("images/tfidf_sparse.npz")
    image_id_dict = np.load("images/id_dict.npy").item()
    image_col_dict = np.load("images/col_dict.npy").item()
    image_row_dict = np.load("images/row_dict.npy").item()
    location_tf = sparse.load_npz("locations/tf_sparse.npz")
    location_df = sparse.load_npz("locations/df_sparse.npz")
    location_tfidf = sparse.load_npz("locations/tfidf_sparse.npz")
    location_id_dict = np.load("locations/id_dict.npy").item()
    location_col_dict = np.load("locations/col_dict.npy").item()
    location_row_dict = np.load("locations/row_dict.npy").item()
    location_name_dict = np.load("locations/location_name_dict.npy").item()
    location_internal_name_dict = np.load("locations/location_internal_name_dict.npy").item()

def cosine_similarity(v1, v2):
    """
    Calculates the cosine similarity between two vectors and returns the similarity value along with the top 3 contributing terms
    """
    dot_product = 0
    similarity_dict = {}

    #calculate dot product
    for col_index in v1.indices:
        if(col_index in v1.indices):
            dot_product += v1[0,col_index] * v2[0,col_index]
            similarity_dict[col_index] = v1[0,col_index] * v2[0,col_index]
    top_items = heapq.nlargest(3, similarity_dict, key=similarity_dict.get) #get top 3 contributing terms

    #calculate magnitude of first vector
    v1_mag = 0
    for col_index in v1.indices:
        v1_mag += v1[0,col_index] * v1[0,col_index]
    v1_mag = np.sqrt(v1_mag)

    #calculate magnitude of second vector
    v2_mag = 0
    for col_index in v2.indices:
        v2_mag += v2[0,col_index] * v2[0,col_index]
    v2_mag = np.sqrt(v2_mag)

    return (dot_product / (v1_mag*v2_mag)), top_items

def calculate_similarity(input_id, model, id_dict, col_dict, row_dict, k):
    """
    Calculates similarity based on textual descriptors of the given input image/user/location
    """
    query_row = model.getrow(row_dict[input_id]) #get input vector
    similarities = {} #dictionary to store similarity between input vector and vector in the database
    similarity_top_terms = {} #dictionary to store top terms contributing to similarity between input vector and vector in database
    col_term_dict = {value:key for key,value in col_dict.items()} #reversed col_dict dictionary eg. numeric_index->term
    #calculate similarities against all vectors
    for (key,value) in id_dict.items():
        d, top_terms = cosine_similarity(query_row, model.getrow(row_dict[value]))
        similarities[value] = d
        similarity_top_terms[value] = top_terms
    top_similarities = heapq.nlargest(k, similarities, key=similarities.get) #get top k similar items
    result = {} #dictionary to store result
    #build result dictionary
    for index in top_similarities:
        d = {}
        d["score"] = similarities[index]
        d["max_terms"] = [col_term_dict[term] for term in similarity_top_terms[index]]
        result[index] = d
    return result

def load_location_data(filename):
    """
    loads the csv file data for a given location model into a dictionary 
    {"image_id" : [value_1, value_2, ...]}
    """
    data_file = open(filename, 'r', encoding='utf-8')
    matrix = {}
    for line in data_file.readlines():
        values = line.strip().split(',')
        index_name = values[0]
        values = [float(x) for x in values[1:]]
        matrix[index_name] = values
    return matrix

def get_location_image_similarity(image1, image2, model):
    """
    calculates similarity between 2 images when locations are being compared based on given model
    
    Model: Similarity measure
    CM : L1 Norm distance
    CM3x3 : average of L1-Norm distances between 3x3 sub images
    CN : Euclidean distance
    CN 3x3 : average of Euclidean distances between 3x3 sub images
    CSD : Euclidean distance
    GLRLM : Euclidean distance
    GLRLM 3x3 : average of Euclidean distances between 3x3 sub images
    HOG : Euclidean distance
    LBP : Euclidean distance
    LBP 3x3: average of Euclidean distances between 3x3 sub images
    """
    if(model == "cm"):
        return np.absolute(image1-image2).sum() #calculate l1 norm distance
    elif(model == "cm3x3"):
        s = 0
        diff_image = image1 - image2
        for i in range(0, diff_image.shape[0],9): #iterate over subimages and calculate similarity
            s += np.absolute(diff_image[i:i+9]).sum()
        s /= 9
        return s
    elif(model == "cn" or model == "csd" or model == "glrlm" or model == "hog" or model == "lbp"):
        return np.linalg.norm(image1-image2) #calculate euclidean distance
    elif(model == "cn3x3"):
        diff_image = image1-image2
        s = 0
        for i in range(0, diff_image.shape[0], 11):
            s+= np.linalg.norm(diff_image[i:i+11])
        s /= 9
        return s
    elif(model == "glrlm3x3"):
        diff_image = image1-image2
        s = 0
        for i in range(0, diff_image.shape[0], 44):
            s+= np.linalg.norm(diff_image[i:i+44])
        s /= 9
        return s
    elif(model == "lbp3x3"):
        diff_image = image1-image2
        s = 0
        for i in range(0, diff_image.shape[0], 16):
            s+= np.linalg.norm(diff_image[i:i+16])
        s /= 9
        return s

def compare_locations(location1, location2, model):
    """
    compares all pairs of images between 2 locations based on the given model
    returns similarity of 2 locations based on average of distance among all image pairs and top contributing image pairs
    """
    total_dist = 0
    n = 0
    similarities = {} #dictionary to store similarity between a pair of images
    for location1_key in location1.keys():
        for location2_key in location2.keys():
            d = get_location_image_similarity(np.array(location1[location1_key]), np.array(location2[location2_key]), model)
            similarities[(location1_key, location2_key)] = d
            total_dist += d
            n += 1
    top_image_pairs = heapq.nsmallest(3, similarities, key=similarities.get) #get top contributing image pairs
    return (total_dist/n), top_image_pairs


def calculate_location_similarity(input_id, model, k):
    """
    calculates the similarity of a given location with all other locations in the database and returns top k locations
    along with top contributing image pairs
    """
    global location_internal_name_dict
    location_name = location_internal_name_dict[input_id] #get internal location name from id eg. 1->angel_of_the_north
    filename = "../descvis/descvis/img/"+location_name+" "+model+".csv" #get corresponding data file name
    query_location = load_location_data(filename) #load the location data
    similarities = {} #dictionary to store similarity between input location and location in database
    top_images = {} #dictionary to store list of image pairs with highest contribution to similarity
    for key in location_internal_name_dict.keys():
        location_name = location_internal_name_dict[key]
        filename = "../descvis/descvis/img/"+location_name+" "+model+".csv"
        test_location = load_location_data(filename)
        sim, top_image_pairs = compare_locations(query_location, test_location, model)
        similarities[location_name] = sim
        top_images[location_name] = top_image_pairs
    top_locations = heapq.nsmallest(k, similarities, key=similarities.get) #get top matching locations
    result = {} #dictionary to store results
    #build result dictionary
    for location in top_locations:
        d = {}
        d["score"] = similarities[location]
        d["max_terms"] = top_images[location] #top_image_pairs
        result[location] = d
    return result


def multiprocess_compare_location(query_location_name, test_location_name, similarity_model_contribution):
    """
    Function to compare 2 locations based on a list of models
    Updates the process manager dictionary with similarity value between the locations and 
    individual model contributions towards similarity
    """
    models = ['CM','CM3x3','CN','CN3x3','CSD','GLRLM','GLRLM3x3','HOG','LBP','LBP3x3']
    model_contributions = {} #dictionary to store similarity contribution of each model
    for model in models:
        model = model.lower()
        filename = "../descvis/descvis/img/"+query_location_name+" "+model+".csv"
        query_location = load_location_data(filename) #load query location data
        filename = "../descvis/descvis/img/"+test_location_name+" "+model+".csv"
        test_location = load_location_data(filename) #load database location data
        sim, top_image_pairs = compare_locations(query_location, test_location, model) #calculate similarity for given model
        model_contributions[model] = sim
    similarity_model_contribution[test_location_name] = model_contributions

def scale_model_similarities(similarity_model_contribution):
    """
    Scales down the similarities calculated among all locations for all models
    scaling is done by dividing the similarity of 2 locations for a given model by dividing by the maximum value for the particular model
    """
    model_max_values = {}
    #get maximum values for each model
    for location in similarity_model_contribution.keys():
        for model in similarity_model_contribution[location].keys():
            if(model in model_max_values.keys()):
                if(similarity_model_contribution[location][model] > model_max_values[model]):
                    model_max_values[model] = similarity_model_contribution[location][model]
            else:
                model_max_values[model] = similarity_model_contribution[location][model]

    #scale down values
    for location in similarity_model_contribution.keys():
        for model in similarity_model_contribution[location]:
            similarity_model_contribution[location][model] /= model_max_values[model]

    return similarity_model_contribution

def multiprocess_task5(input_id, k):
    """
    Compares 2 locations on a separate process 
    """
    global location_internal_name_dict
    query_location_name = location_internal_name_dict[input_id]
    manager = multiprocessing.Manager()
    similarity_model_contribution = manager.dict() #dictionary to store similarity contribution of various models when comparing 2 locations 
    pool = multiprocessing.Pool()
    for key in location_internal_name_dict.keys():
        test_location_name = location_internal_name_dict[key]
        #create processes for comparing a pair of locations
        pool.apply_async(multiprocess_compare_location, args=(query_location_name, test_location_name, similarity_model_contribution))
    pool.close()
    pool.join() #wait for all comparisons to complete

    similarity_model_contribution = dict(similarity_model_contribution)
    similarity_model_contribution = scale_model_similarities(similarity_model_contribution)

    # Compute average of all models
    similarity = {}
    for location in similarity_model_contribution:
        s = 0
        for model in similarity_model_contribution[location]:
            s += similarity_model_contribution[location][model]
        s /= 10
        similarity[location] = s

    top_locations = heapq.nsmallest(k, similarity, key=lambda k: similarity[k]) #get top similar locations
    result = {}
    #build result dictionary
    for location in top_locations[:k]:
        d = {}
        d["score"] = similarity[location]
        d["max_terms"] = similarity_model_contribution[location]
        result[location] = d
    return result

def main():
    while True:
        print("Choose task")
        print("1. Task 1")
        print("2. Task 2")
        print("3. Task 3")
        print("4. Task 4")
        print("5. Task 5")
        print("6. Exit")
        task_type = input()
        task_type = int(task_type)
        if(task_type <= 4):
            values = input("Enter id, model, k\n")
            values = values.strip().split(' ')
            k = int(values[-1])
            model = values[-2].lower()
            input_id = values[0]
        elif(task_type == 5):
            values = input("Enter id, k\n")
            values = values.strip().split(' ')
            input_id = values[0]
            k = int(values[1])

        result = None
        if(task_type == 1):
            if(model == "tf"):
                result = calculate_similarity(input_id, user_tf, user_id_dict, user_col_dict, user_row_dict, k)
            elif(model == "df"):
                result = calculate_similarity(input_id, user_df, user_id_dict, user_col_dict, user_row_dict, k)
            elif(model == "tf-idf"):
                result = calculate_similarity(input_id, user_tfidf, user_id_dict, user_col_dict, user_row_dict, k)
        elif(task_type == 2):
            if(model == "tf"):
                result = calculate_similarity(input_id, image_tf, image_id_dict, image_col_dict, image_row_dict, k)
            elif(model == "df"):
                result = calculate_similarity(input_id, image_df, image_id_dict, image_col_dict, image_row_dict, k)
            elif(model == "tf-idf"):
                result = calculate_similarity(input_id, image_tfidf,image_id_dict, image_col_dict, image_row_dict, k)
        elif(task_type == 3):
            input_id = location_name_dict[int(input_id)]
            if(model == "tf"):
                result = calculate_similarity(input_id, location_tf, location_id_dict, location_col_dict, location_row_dict, k)
            elif(model == "df"):
                result = calculate_similarity(input_id, location_df, location_id_dict, location_col_dict, location_row_dict, k)
            elif(model == "tf-idf"):
                result = calculate_similarity(input_id, location_tfidf, location_id_dict, location_col_dict, location_row_dict, k)
        elif(task_type == 4):
            result = calculate_location_similarity(int(input_id), model, k)
        elif(task_type == 5):
            result = multiprocess_task5(int(input_id), k)
        else:
            break

        ordered_keys = None
        if(task_type <= 3):
            ordered_keys = sorted(result.keys(), key = lambda k: result[k]["score"], reverse=True)
        else:
            ordered_keys = sorted(result.keys(), key = lambda k: result[k]["score"], reverse=False)
        for key in ordered_keys:
            print(key, result[key]["score"], result[key]["max_terms"])

if __name__ == "__main__":
    load_data()
    main()     