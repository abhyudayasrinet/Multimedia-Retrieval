import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances


model_indices = {
    'CM': (0, 9),
    'CM3x3': (9, 90),
    'CN': (90, 101),
    'CN3x3': (101, 200),
    'CSD': (200, 264),
    'GLRLM': (264, 308),
    'GLRLM3x3': (308,  704),
    'HOG': (704, 785),
    'LBP': (785, 801),
    'LBP3x3': (801, 945)
}

index_to_image_dict = {}
image_to_index_dict = {}
images = []
image_count = 0


def update_image_index_dict(image_ids):
    """
    Updates the index to image id dictionary
    :param image_ids:
    :return: Nothing
    """
    global image_count
    global images
    for image_id in image_ids:
        image_to_index_dict[image_id] = image_count
        index_to_image_dict[image_count] = image_id
        images = np.append(images, [str(image_id)])
        image_count = image_count + 1


def CM(imgsvec1, imgsvec2):
    # calculate the pair wise manhattan distance between two vector matrices
    allDis = manhattan_distances(imgsvec1, imgsvec2)
    return allDis


def CM3x3(imgsvec1, imgsvec2):
    # number of features of the CM model
    features = 9

    start = 0

    # initialize an empty matrix
    sumVec = np.zeros((imgsvec1.shape[0], imgsvec2.shape[0]))

    # loop over the 9 3x3 blocks
    for i in range(1, 10):
        end = i * features

        # get the vector of this block
        img1block = imgsvec1.iloc[:, start:end]
        img2block = imgsvec2.iloc[:, start:end]

        # distance b/w block of image 1 and image 2
        blockDis = manhattan_distances(img1block, img2block)

        sumVec = sumVec + blockDis

        # start index of next block
        start = end

    # average of distance between all 9 blocks of two images
    sumVec = sumVec / 9
    return sumVec


def CN(imgsvec1, imgsvec2):
    # calculate the pair wise euclidean distance between two vector matrices
    allDis = euclidean_distances(imgsvec1, imgsvec2)
    return allDis


def CN3x3(imgsvec1, imgsvec2):
    # number of features of the CN model
    features = 11

    start = 0

    # initialize a matrix to store the distances of two images
    sumVec = np.zeros((imgsvec1.shape[0], imgsvec2.shape[0]))

    for i in range(1, 10):
        end = i * features

        # distance b/w block of image 1 and image 2
        img1block = imgsvec1.iloc[:, start:end]
        img2block = imgsvec2.iloc[:, start:end]

        # distance b/w block of image 1 and image 2
        blockDis = euclidean_distances(img1block, img2block)

        sumVec = sumVec + blockDis

        # start index of the next image block
        start = end

    # average of distance between all 9 blocks of two images
    sumVec = sumVec / 9
    return sumVec


def CSD(imgsvec1, imgsvec2):
    # calculate the pair wise euclidean distance between two vector matrices
    allDis = euclidean_distances(imgsvec1, imgsvec2)
    return allDis


def GLRLM(imgsvec1, imgsvec2):
    # calculate the pair wise euclidean distance between two vector matrices
    allDis = euclidean_distances(imgsvec1, imgsvec2)
    return allDis


def GLRLM3x3(imgsvec1, imgsvec2):
    # number of features of the CN model
    features = 44

    start = 0

    # initialize a matrix to store the distances of two images
    sumVec = np.zeros((imgsvec1.shape[0], imgsvec2.shape[0]))

    # loop over the 9 3x3 blocks
    for i in range(1, 10):
        end = i * features

        # get the vector of this block
        img1block = imgsvec1.iloc[:, start:end]
        img2block = imgsvec2.iloc[:, start:end]

        # distance b/w block of image 1 and image 2
        blockDis = euclidean_distances(img1block, img2block)

        sumVec = sumVec + blockDis

        # start index of the next image block
        start = end

    # average of distance between all 9 blocks of two images
    sumVec = sumVec / 9
    return sumVec


def HOG(imgsvec1, imgsvec2):
    # calculate the pair wise euclidean distance between two vector matrices
    allDis = euclidean_distances(imgsvec1, imgsvec2)
    return allDis


def LBP(imgsvec1, imgsvec2):
    # calculate the pair wise euclidean distance between two vector matrices
    allDis = euclidean_distances(imgsvec1, imgsvec2)
    return allDis


def LBP3x3(imgsvec1, imgsvec2):
    # number of features of this LBP model
    features = 16

    start = 0

    sumVec = np.zeros((imgsvec1.shape[0], imgsvec2.shape[0]))

    # loop over the 9 3x3 blocks
    for i in range(1, 10):
        end = i * features

        # get the vector of this block
        img1block = imgsvec1.iloc[:, start:end]
        img2block = imgsvec2.iloc[:, start:end]

        # calculate euclidean distance for the blocks
        blockDis = euclidean_distances(img1block, img2block)

        sumVec = sumVec + blockDis

        # start index of the next image block
        start = end

    # average of distance between all 9 blocks of two images
    sumVec = sumVec / 9
    return sumVec


modelsFuncDict = {'CM': CM, 'CM3x3': CM3x3, 'CN': CN, 'CN3x3': CN3x3, 'CSD': CSD, 'GLRLM': GLRLM,
                      'GLRLM3x3': GLRLM3x3,
                      'HOG': HOG, 'LBP': LBP, 'LBP3x3': LBP3x3}


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


def calculate_similarity_matrix(location_count):
    """
    Calculates the similarity matrix for all images.
    :param location_count: Total Number of locations
    :return: Returns the dense image-image distance matrix.
    """
    dfs = []
    for i in range(1, location_count + 1):
        merged_input_df, image_ids = get_merged_data_frame_per_location(i)
        dfs.append(merged_input_df)
        update_image_index_dict(image_ids)

    all_images = pd.concat(dfs, axis=0)
    all_images = normalize_data(all_images)

    num_of_images = len(all_images)
    print(num_of_images, ' images found')

    sim_matrix = np.zeros((num_of_images, num_of_images))

    for model in modelsFuncDict.keys():
        start = model_indices[model][0]
        end = model_indices[model][1]
        input = all_images.iloc[:, start:end]
        sim_matrix += modelsFuncDict[model](input, input)
        print(model, ' done')

    sim_matrix = sim_matrix / 10
    return sim_matrix


if __name__ == '__main__':
    k = input('Enter k: ')
    k = int(k)

    dense_sim_matrix = calculate_similarity_matrix(get_number_of_locations())

    sorted_index_sim_matrix = np.argsort(dense_sim_matrix, axis=1)

    for row, image in enumerate(sorted_index_sim_matrix):
        count_k = 0
        for col in image:
            if count_k < k and row != col:
                dense_sim_matrix[row][col] = dense_sim_matrix[row][col]
                count_k = count_k + 1
            else:
                dense_sim_matrix[row][col] = 0

    df = pd.DataFrame(dense_sim_matrix)
    df.to_csv('ImageSimilarities.csv')

    df = pd.DataFrame(images)
    df.to_csv('IndexToImage.csv')
