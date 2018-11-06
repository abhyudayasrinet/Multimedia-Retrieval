import xml.etree.ElementTree as ET
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation


def get_number_of_locations():
    """
    Returns the number of locations in the devset_topics.xml file.
    :return: Total number of locations
    """
    count = 0
    tree = ET.parse('./devset_topics.xml')
    root = tree.getroot()
    for item in root.findall('./topic'):
        count = count + 1
    return count


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
    for model in models:
        input_file = './descvis/img/' + get_location_from_id(id) + ' ' + model + '.csv'
        df = pd.read_csv(input_file, header=None)
        df = df.drop(df.columns[0], axis=1)
        if min_rows > df.shape[0]:
            min_rows = df.shape[0]
        dfs.append(df)

    merged = pd.concat(dfs, axis=1)
    merged = merged[:min_rows]

    return merged


def dimentionality_reduction(algo, k):
    """
    Performs dimensionality reduction based on the algo and finds k latent features
    and returns the new k dimensional space.
    :param algo: Algorithm to use(SVD, PCA or LDA) Default: PCA
    :param k: Number of latent features to find
    :return: New K Dimensional Space
    """
    algo = algo.upper();
    if algo == 'SVD':
        return TruncatedSVD(n_components=int(k))
    elif algo == 'LDA':
        return LatentDirichletAllocation(n_components=int(k))
    else:
        return PCA(n_components=int(k))


def normalize_data(data_frame):
    """
    Normalizes the data in the matrix to values between 0 and 1 for each feature.
    :param data_frame: Matrix to be normalized.
    :return: Normalized Matrix
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data_frame)
    return pd.DataFrame(x_scaled)


def calculate_similarity_measures(input_matrix, dimensional_space, location_count):
    """
    Calculates and returns a list of similarity values for a given location compared to all the other locations.
    :param input_matrix: Input Location Matrix
    :param dimensional_space: The K dimensional space for Input Location Matrix
    :param location_count: Total number of locations
    :return: List of objects having a similarity value and the name of the location.
    """
    similarity_measures = []
    for i in range(1, location_count+1):
        merged_df = get_merged_data_frame_per_location(str(i))
        merged_df = normalize_data(merged_df)
        compared_location_transformed = dimensional_space.transform(merged_df)
        similarity_matrix = euclidean_distances(input_matrix, compared_location_transformed)
        similarity_measures.append({'value': similarity_matrix.mean(), 'location': get_location_from_id(str(i))})

    return similarity_measures


if __name__ == '__main__':
    location_id = input('Enter Location ID: ')
    k = input('Enter k: ')
    algo = input('Enter Dimentionality reduction algorithm: ')

    new_dimensional_space = dimentionality_reduction(algo, k)

    merged_input_df = get_merged_data_frame_per_location(location_id)
    merged_input_df = normalize_data(merged_input_df)
    new_dimensional_space.fit(merged_input_df)
    input_location_transformed = new_dimensional_space.transform(merged_input_df)

    similarity_measures = calculate_similarity_measures(input_location_transformed, new_dimensional_space, get_number_of_locations())
    similarity_measures.sort(key=lambda x: x['value'])

    print('5 most related locations: ')
    for i in range(5):
        print(similarity_measures[i]['location'], ', similarity = ', similarity_measures[i]['value'])
