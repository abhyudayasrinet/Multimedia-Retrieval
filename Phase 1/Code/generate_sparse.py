import numpy as np
from scipy import sparse

# Following script converts the textual descriptor files to sparse matrices for users/images/locations and stores them in the respective fodlers

user_data = r"../desctxt/desctxt/devset_textTermsPerUser.txt"
image_data = r"../desctxt/desctxt/devset_textTermsPerImage.txt"
location_data = r"../desctxt/desctxt/devset_textTermsPerPOI.txt" 

data_files = [user_data, image_data, location_data]
for data_file in data_files:
    filename = data_file
    data_file = open(filename, 'r', encoding='utf-8')

    terms = set()
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

    #arrays to store tf,df,tfidf values
    tf_data = []
    df_data = []
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
            tf = float(elements[i+1])
            df = float(elements[i+2])
            tfidf = float(elements[i+3])
            #add term and indices to respective arrays
            col_index = col_dict[term]
            tf_data.append(tf)
            df_data.append(df)
            tfidf_data.append(tfidf)
            row_indices.append(row_index)
            col_indices.append(col_index)
        row_index += 1

    #create sparse matrices from extracted data
    tf = sparse.csr_matrix((tf_data, (row_indices, col_indices)))
    df = sparse.csr_matrix((df_data, (row_indices, col_indices)))
    tfidf = sparse.csr_matrix((tfidf_data, (row_indices, col_indices)))

    #save the created matrices and index mapping to the respective output folder
    output_folder = None
    if filename == user_data:
        output_folder = "users"
    elif filename == image_data:
        output_folder = "images"
    elif filename == location_data:
        output_folder = "locations"

    np.save(output_folder+'/row_dict.npy', row_dict)
    np.save(output_folder+'/col_dict.npy', col_dict)
    np.save(output_folder+'/id_dict.npy', id_dict)
    sparse.save_npz(output_folder+"/tf_sparse.npz", tf)
    sparse.save_npz(output_folder+"/df_sparse.npz", df)
    sparse.save_npz(output_folder+"/tfidf_sparse.npz", tfidf)