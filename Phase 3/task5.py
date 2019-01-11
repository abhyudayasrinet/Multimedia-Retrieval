import numpy as np
import scipy.sparse
import scipy.spatial
import time
from sklearn.metrics.pairwise import euclidean_distances
from common import store_results

vecs = None
index_to_image_dict = None
image_to_index_dict = None


def load_data():

    global vecs, index_to_image_dict, image_to_index_dict

    vecs = scipy.sparse.load_npz("images/image_features_sparse.npz")
    index_to_image_dict = np.load("images/index_to_image_dict.npy").item()
    image_to_index_dict = np.load("images/image_to_index_dict.npy").item()


def generate_hash(planes, input_vec):

    layer_hash = planes.dot(input_vec.transpose())

    hash_list = ["1" if i>0 else "0" for i in layer_hash]

    hash_bit = "".join(hash_list)

    return hash_bit


def get_euclidean_distance(vec1, vec2):

    return euclidean_distances(vec1, vec2)


def get_closet_hash(key, keylist):

    distance_dict = {}

    key_array = [float(c) for c in key]

    for kl in keylist:

        hamm_dis = scipy.spatial.distance.hamming(key_array, [float(c) for c in kl])

        distance_dict[kl] = hamm_dis


    return sorted(distance_dict.items(), key=lambda item: item[1])


def get_closet_members(query_vec, layers, all_layers_planes, fetched_keys):

    other_members = []

    for i, layerDict in enumerate(layers):

        key = generate_hash(all_layers_planes[i], query_vec)

        allkeys = list(layerDict.keys())

        if key in allkeys:
            
            allkeys.remove(key)

        otherkeys = [ke for ke in allkeys if ke not in fetched_keys]

        #find the closet key
        closet_key = get_closet_hash(key, otherkeys)[0][0]

        other_members += layerDict[closet_key]

        fetched_keys.append(closet_key)

    return other_members, fetched_keys



def lsh_index(input_index, input_vec, layers, all_layers_planes):

    for i, layerDict in enumerate(layers):

        key = generate_hash(all_layers_planes[i], input_vec)

        if key not in layerDict:

            layerDict[key] = []

        layerDict[key].append(index_to_image_dict[input_index])


def lsh_query(query_vec, t, layers, all_layers_planes):

    members = []

    result_dict = {}

    fetched_keys = []

    for i, layerDict in enumerate(layers):

        key = generate_hash(all_layers_planes[i], query_vec)

        if key in layerDict:

            members += layerDict[key]

            fetched_keys.append(key)


    all_members = len(members)
    members = list(set(members))
    unique_members = len(members)

    
    while unique_members < t:

        closet_members, fetched_keys = get_closet_members(query_vec, layers, all_layers_planes, fetched_keys)

        members += closet_members

        #re-calculate the number of candidate members

        all_members = len(members)
        members = list(set(members))
        unique_members = len(members)


    for member in members:

        member_vec = vecs[image_to_index_dict[member]]

        euc_dis = get_euclidean_distance(member_vec, query_vec)

        result_dict[member] = euc_dis


    sorted_dict = sorted(result_dict.items(), key=lambda item: item[1])

    return sorted_dict[:t], all_members, unique_members




if __name__ == "__main__":

    #input for number of layers and number of hash functions per layer

    L = int(input("Enter L:\n"))
    k = int(input("Enter k:\n"))

    start = time.time()

    #Part a

    #load the object-feature matrix and data 

    load_data()

    vec_size = vecs.shape[1]

    #object to store hash values of all layers

    layers = [{} for _ in range(L)]

    # generate hyperplanes to hash the input points

    all_layers_planes = []

    for i in range(L):

        # for every layer generate k hyper planes of the size of the input vector

        planes = np.random.randn(k, vec_size)

        all_layers_planes.append(scipy.sparse.csr_matrix(planes))

    #index all the input points

    for i in range(vecs.shape[0]):

        lsh_index(i, vecs[i], layers, all_layers_planes)


    print ("\nIndex structure created.\n")

    #print ("\nTime Taken: ", (time.time()-start))


    #Part b

    ans = "y"

    while ans == "y":
    
        query = input("\nEnter the query image id:\n")
        t = int(input("Enter t:\n"))

        query_index = image_to_index_dict[query]

        query_vec = vecs[query_index]


        results = lsh_query(query_vec, t, layers, all_layers_planes)

        images = results[0]
        all_members = results[1]
        unique_members = results[2]

        imageids = []

        for img in images:

            imageids.append(img[0])


        print ("\nImages: ", imageids)
        print ("Number of unique images considered: ", unique_members)
        print ("Number of overall images: ", all_members)

        mapping = {
            '0': 'Input Images',
            '1': 'T Similar Images'
        }

        store_results([[query],imageids], "5", mapping, True, 'Locality Sensitive Hashing')

        ans = input("\nAnother query?\n")

