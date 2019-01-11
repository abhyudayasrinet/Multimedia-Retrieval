import numpy as np
import pandas as pd
import csv
import time
import common

"""Function to take user input when running as Task4
2 user inputs are taken :
    1. k : Number of top relevant images to be shown to the user
    2. input_images : As set of 3 input image IDs taken from the user for PPR Algorithm
"""


def userInput():
    print('Please enter a value of k:')
    input_k = int(input())

    input_images = []
    print('Please enter input image IDs :')
    for _ in range(0, 3):
        input_images.append(input())

    return input_k, input_images


"""
Main function to calculate the Page Rank of all the nodes of the graph.
This function takes 4 optional input parameters.
    1. graph - Optional parameter of type DataFrame which takes an adjacency matrix of the graph.
    2. input_images - Optional parameter which accepts a list of input images. If input images are not provided, it is taken as input from user.
    3. beta - Damping factor.   Default Value : 0.85
    4. epsilon : Error Threshold of Page Rank scores. Default Value : 0.000001
"""


def pageRank(graph=None, input_images=None, beta=0.85, epsilon=0.000001):
    mapping = {
        '0': 'Input Images',
        '1': 'K Dominant Images',
    }
    # If graph is not provided, it loads the ImageSimilarities.csv file
    if graph is None:
        print('Loading Image-Image Similarity Graph...')

        graph = pd.read_csv("ImageSimilarities.csv", index_col=0).values
        graph[graph.nonzero()] = 1
        graph = pd.DataFrame(graph).T

    nodes = len(graph)

    # Dictionaries mapping index number of adjacency matrix to the image IDs and vice-a-versa
    index_img_dict = dict(csv.reader(open('IndexToImage.csv', 'r')))
    img_index_dict = dict([[v, k] for k, v in index_img_dict.items()])

    M = graph / np.sum(graph, axis=0)

    # Initializing Teleportation matrix and Page Rank Scores with Zeros for all images
    teleportation_matrix = np.zeros(nodes)
    pageRankScores = np.zeros(nodes)

    # Takes user input if input_images are not provided
    if input_images is None:
        input_k, input_images = userInput()
        isTask6 = False
    else:
        isTask6 = True

    if isTask6 is False:
        print('Calculating Personalized PageRank Score with a Damping Factor of ' + str(beta) + '...')

    # Updating Teleportation and Page Rank Score Matrices with 1/num_of_input images for the input images.
    for image_id in input_images:
        teleportation_matrix[int(img_index_dict[image_id])] = 1 / len(input_images)
        pageRankScores[int(img_index_dict[image_id])] = 1 / len(input_images)

    # Calculating Page Rank Scores
    while True:
        oldPageRankScores = pageRankScores
        pageRankScores = (beta * np.dot(M, pageRankScores)) + ((1 - beta) * teleportation_matrix)
        if np.linalg.norm(pageRankScores - oldPageRankScores) < epsilon:
            break

    pageRankScores[np.isnan(pageRankScores)] = 0

    # Normalizing Page Rank Scores
    pageRankScores = pageRankScores / sum(pageRankScores)

    # Displaying output for Task 4
    if isTask6 is not True:
        output = []
        for index in reversed(np.argsort(pageRankScores)[-input_k:]):
            output.append(index_img_dict[str(index)])
        # Function call to display results in Web Page.
        print(output)
        common.store_results([input_images, output], '4', mapping, True, 'Personalized Page Rank')
    # Returning results for Task 6
    else:
        return pageRankScores


if __name__ == '__main__':
    start_time = time.time()
    pageRank(beta=0.5)
    end_time = time.time()
    print('Total Time : ', end_time - start_time)
