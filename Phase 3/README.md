Multimedia and Web Databases Project - Phase 3
----------------------------------------------

Description
-----------
The goal of this phase of the project is to identify the clusters, classes and most dominant images among the given images based on an image-image similarity graph using various clustering, partitioning, ranking, and classification algorithms and visualize the results.

The dataset consists of data about certain number of locations, their corresponding images and the users who uploaded the images. Visual descriptors of all the images are also given based on 10 models - CN, CM, CN3x3, CM3x3, HOG, LBP, LBP3x3, GLRLM, GLRLM3x3, CSD grouped by locations. These 10 descriptors are used to create the above mentioned image-image similarity graph.


Getting Started
---------------
The below instructions will help you set the project up and running on your local Windows machine for development and testing purposes.

Prerequisites
-------------
Before running and testing the programs included in this project, follow the below steps to set up the environment.

**Installing Python v3.7 for Windows**
1. Open a web browser and visit https://www.python.org/downloads/
2. Download the latest stable version(v3.7.0 at the time of development) .exe file.
3. Run the .exe file once downloaded and follow the steps in the wizard to install.

**Python Libraries required to run the Programs**
1. pandas
2. numpy
3. bs4
4. lxml
5. sklearn
6. scipy
7. tqdm


Enter the below command in command prompt to install the libraries:
pip install <library_name>

Ex: pip install scipy

Note: Open command prompt as administrator if you have troubles installing the libraries.

**Development/Test Datasets**
1. Open any Web Browser and visit http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/
2. Click on devset/testset link under Resources tab and download the following files required for testing and development purposes.
  a. descvis.zip
  b. img.zip
  c. devset_topics.xml


3. Align the folders and the python files according to the paths shown below:
  .\img\          											#This is the img folder which is originally present inside the descvis folder
  .\devset_topics.xml
  .\task1.py
  .\task2_spectral_cl.py
  .\task2_conf_cl.py
  .\task3.py
  .\task4.py
  .\generate_data.py
  .\task5.py
  .\task6_1.py
  .\task6_2.py


Running the Tests
-----------------
Now that we have done the environment setup, refer to the below detailed program description along with the command to execute the respective program and follow the on screen instructions to get the output.

Note: Steps 1 and 2 must be run before running any of the tasks.

1. task1.py
This program uses the visual descriptors for all the images to create a image-image similarity graph with each node having k-outgoing edges to k-most similar images. The output of this task is a csv file containing the adjacency matrix.

Command to run:
python task1.py

2. task2_spectral_cl.py
This program takes 'c' as the user input representing the number of clusters and the image-image similarity graph created as an output of task1. It then applies Spectral Clustering algorithm to identify 'c' clusters among all the images based on the input graph. The output of the task is a set of clusters.

Command to run:
python task2_spectral_cl.py

3. task2_conf_cl.py
This program takes 'c' as the user input representing the n umber of clusters and the image-image similarity graph created as an output of task1. It then applies Confidence Clustering algorithm to identify 'c' clusters among all the images based on the input graph. The output of the task is a set of clusters.

Command to run:
python task2_conf_cl.py

4. task3.py
This program takes as input the image-image similarity graph created as output of task1, a user supplied value of 'k' to find the dominant 'k' images in the given graph using Page Rank algorithm.

Command to run:
python task3.py

5. task4.py
This program takes as input the image-image similarity graph created as output of task1, a user supplied value of 'k', and 3 user supplied image_ids to find the most relevant 'k' images to the user specified images in the given graph using Personalized Page Rank algorithm.

Command to run:
python task4.py

6. generate_data.py
This program generates the sparse image-feature matrix, index-image id and image id-index mapping required in task 5.

Command to run:
python generate_data.py

7. task5.py
This task takes the number of layers 'L' and the number of hashes per layer 'k' as the user input and uses the visual descriptors of all the images to create the Locality Sensitive Hashing Tool. It then takes in the the query image id and the number of similar images to return 't' as the input. The output is 't' similar images to the query image.

Command to run:
python task5.py

8. task6_1.py
This task takes training file as input, k and output of task1 as input and classify test images using K-neareast neighbour algorithms
Command to run:
python task6_1.py


9. task6_2.py
This task takes as input the image-image similarity graph created as output of task1 and a set of image label pairs as input which it uses as training data to classify the remaining images of the database using Personalized Page Rank algorithm.

Command to run:
python task6_2.py


Authors
-------
1. Abhyudaya Srinet
2. Adhiraj Tikku
3. Anjali Singh
4. Darshan Dagly
5. Laveena Sachdeva
6. Vatsal Sodha
