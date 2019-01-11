from task4 import pageRank
import time
import csv
import pandas as pd
import common


def main():
    print('Loading Image-Image Similarity Matrix...')

    # Loading the ImageSimilarities.csv file
    graph = pd.read_csv("ImageSimilarities.csv", index_col=0).values
    graph[graph.nonzero()] = 1
    graph = pd.DataFrame(graph).T

    # Dictionary mapping index number of adjacency matrix to the image IDs
    index_img_dict = dict(csv.reader(open('IndexToImage.csv', 'r')))

    labels = set()

    # Reads user input file
    input_image_label_pair = []
    with open('task6_sample_input.txt', 'r') as file:
        for line in file:
            image_id = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            labels.add(label)
            input_image_label_pair.append([image_id, label])

    # Calculating Personalized Page Rank for each label once.
    output = []
    label_dict = dict()
    count = 0
    for label in labels:
        label_dict[str(count)] = label
        count += 1
        print('Calculating Personalised Page Rank for label : ' + label)
        input_images = [item[0] for item in input_image_label_pair if item[1] == label]
        output.append([label, pageRank(graph, input_images, beta=0.5).tolist()])
    # print(label_dict)

    # Assigning label to each image based on the highest Page Rank score value
    image_labels = []
    for i in range(0, len(output[0][1])):
        compare = []
        for item in output:
            compare.append([item[0], i, item[1][i]])
        image_labels.append(sorted(compare, reverse=True, key=lambda x: x[2])[0])

    # Grouping images for each labels
    final_output = dict()
    for elem in image_labels:
        if elem[0] not in final_output:
            final_output[elem[0]] = []
        final_output[elem[0]].append(index_img_dict[str(elem[1])])
    print(final_output)

    final_final_output = []

    for k, v in label_dict.items():
        final_final_output.append(final_output[v])
    print(final_final_output)

    # Function call to display results in Web Page.
    common.store_results(final_final_output, '6.2', label_dict, False, 'Personalized Page Rank')


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Time : ', end_time - start_time)
