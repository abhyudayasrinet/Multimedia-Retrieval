import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy
import os

index_to_image_dict = {}
image_to_index_dict = {}
images = []
image_count = 0

output_folder = "images"

def get_location_from_id(id):
    """
    Returns the location name for a given location ID
    :param id: Location ID
    :return: Location Name
    """
    tree = ET.parse('./devset_topics.xml')
    root = tree.getroot()
    for item in root.findall('./topic'):
        if id == item[0].text:
            return item[1].text


def get_merged_data_frame_per_location(id):
    """
    Concatenates the features for a location across all the 10 visual
    descriptor models into a single Location-Feature matrix
    :param id: Location ID
    :return: Location-Feature Matrix
    """
    dfs = []
    models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
    min_rows = float('inf')
    image_ids = []
    flag = 0
    for model in models:
        input_file = './img/' + get_location_from_id(str(id)) + ' ' + model + '.csv'
        df = pd.read_csv(input_file, header=None)
        if flag == 0:
            image_ids = df.ix[:, 0].values
            flag = 1
        df = df.drop(df.columns[0], axis=1)
        if min_rows > df.shape[0]:
            min_rows = df.shape[0]
        dfs.append(df)

    merged = pd.concat(dfs, axis=1)
    merged = merged[:min_rows]

    return merged, image_ids

def normalize_data(data_frame):
    """
    Normalizes the data in the matrix to values between 0 and 1 for each feature.
    :param data_frame: Matrix to be normalized.
    :return: Normalized Matrix
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data_frame)
    return pd.DataFrame(x_scaled)


def update_image_index_dict(image_ids):
    global image_count
    global images
    for image_id in image_ids:
        image_to_index_dict[str(image_id)] = image_count
        index_to_image_dict[image_count] = str(image_id)
        images = np.append(images, [image_id])
        image_count = image_count + 1


dfs = []

for i in range(1, 31):
    merged_input_df, image_ids = get_merged_data_frame_per_location(i)
    merged_input_df = normalize_data(merged_input_df)
    dfs.append(merged_input_df)
    update_image_index_dict(image_ids)

all_images = pd.concat(dfs, axis=0)

mat = scipy.sparse.csr_matrix(all_images.values)

#save the sparse matrix and image-index and index-image data objects to a file

if not os.path.exists(output_folder):

	os.makedirs(output_folder)
	np.save(output_folder+'/index_to_image_dict.npy', index_to_image_dict)
	np.save(output_folder+'/image_to_index_dict.npy', image_to_index_dict)
	scipy.sparse.save_npz(output_folder+"/image_features_sparse.npz", mat)