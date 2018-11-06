from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np
from task6_helper import *

# User Input for k
input_k = int(input('Enter a value for k:'))

# Opening Textual Feature Data file
location_data = '.\desctxt\devset_textTermsPerPOI.txt'
data_file = open(location_data, 'r', encoding='utf-8')

# Creating a set of all unique terms stored in ascending order
terms = set()
for line in data_file.readlines():
    quote_index = line.index("\"")
    elements = line[quote_index:].strip().split(' ')  # skip the name of the location
    for i in range(0, len(elements), 4):  # iterate in steps of 4 to get terms
        terms.add(elements[i][1:-1])  # remove quotes("xyz")
terms = list(terms)
terms.sort()

# arrays to store tfidf values
tfidf_data = []

# array to coordinate indices for sparse matrix
row_indices = []
col_indices = []
row_index = 0

# dictionaries to store mapping of value=>numeric_index of sparse matrix
row_dict = {}  # id->numeric_index
id_dict = {}  # numeric_index->id
col_dict = {term: index for index, term in enumerate(terms)}  # term->numeric_index

# Reading the location data tf-idf values and creating a 2D matrix to store the data
data_file.seek(0)
for line in data_file.readlines():
    quote_index = line.index("\"")
    index_name = line[:quote_index].strip()
    elements = line[quote_index:].strip().split(' ')
    row_dict[index_name] = row_index
    id_dict[row_index] = index_name
    for i in range(0, len(elements), 4):
        # extract each term and their tfidf values
        term = elements[i][1:-1]
        tfidf = float(elements[i + 3])

        # add term and indices to respective arrays
        col_index = col_dict[term]
        tfidf_data.append(tfidf)
        row_indices.append(row_index)
        col_indices.append(col_index)
    row_index += 1

# create sparse matrices from extracted dense 2D matrix
tfidf = sparse.csr_matrix((tfidf_data, (row_indices, col_indices)))

# Calculating the Cosine Similarity using the tf-idf values of each loaction
txtdesc_similarity = cosine_similarity(tfidf)


# Calling task6_helper for calculating location similarity based on Visual Features
visdesc_similarity = calc_sim()

# Taking the average of location similariy based on textual and visual features
cumulative_scores = (txtdesc_similarity + visdesc_similarity) / 2

# Performing SVD on the location x location similarity matrix
transformation = TruncatedSVD(n_components = input_k)
principalComponents = transformation.fit(cumulative_scores)
eigenvectors = principalComponents.components_

# Storing the output of location term weight pairs in CSV files.
f = open("task6_latentsemantics.csv","w")
for i, eigenvector in enumerate(eigenvectors):
    f.write("LATENT SEMANTIC " + str(i+1) + "\n")
    sorted_indices = np.argsort(eigenvector)[::-1]
    for j in sorted_indices:
        f.write(id_dict[j]+",")
    f.write('\n')
    for j in sorted_indices:
        f.write(str(eigenvector[j])+",")
    f.write('\n')