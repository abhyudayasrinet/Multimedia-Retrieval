Phase 2
--------

The goal of this phase of the project is to identify the latent semantics using different dimensionality reduction algorithms and use them to find similarity between objects in the reduced feature space.

The dataset consists of data about certain number of locations, their corresponding images and the users who uploaded the images. The description and tags of the images have been used to find the TF, DF, TF-IDF values for users, images and locations. Visual descriptors of all the images are also given based on 10 models - CN, CM, CN3x3, CM3x3, HOG, LBP, LBP3x3, GLRLM, GLRLM3x3, CSD grouped by locations.


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
7. xmltodict
8. tensorly

Enter the below command in command prompt to install the libraries:
`pip install <library_name>`

Ex: `pip install scipy`

Note: Open command prompt as administrator if you have troubles installing the libraries.

**Development/Test Datasets**
1. Open any Web Browser and visit http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/
2. Click on devset/testset link under Resources tab and download the following files required for testing and development purposes.
  a. desctxt.zip
  b. descvis.zip
  c. img.zip
  d. imgwiki.zip
  e. devset_topics.xml
  f. poiNameCorrespondences.txt

3. Align the folders and the python files according to the paths shown below:
  .\desctxt\devset_textTermsPerPOI.txt
  .\desctxt\devset_textTermsPerUser.txt
  .\desctxt\devset_textTermsPerImage.txt
  .\desctxt\devset_textTermsPerPOI.wFolderNames.txt
  .\descvis\img\
  .\devset_topics.xml
  .\poiNameCorrespondences.txt
  .\generate_sparse.py
  .\parse_xml.py
  .\task1.py
  .\task2.py
  .\task3.py
  .\task4.py
  .\task5.py
  .\task6.py
  .\task6_helper.py
  .\task6_similarity_compute.py
  .\task7.py

Running the Tests
-----------------
Now that we have done the environment setup, refer to the below detailed program description along with the command to execute the respective program and follow the on screen instructions to get the output.

Note: Steps 1 and 2 must be run before running any of the tasks.

1. generate_sparse.py
This program is used to parse the text files in the desctxt folder to create the tfidf sparse matrices and their index mapping dictionaries and saves them into their respective folders named "images", "users" and "locations" as .npy and .npz files. The program adds extra terms to the first document in a vector space if each vector space matrix dimensions do not match.
Command to run:
python generate_sparse.py

2. parse_xml.py
This program parses the devset_topics.xml file to build a dictionary mapping the location names with their respective IDs.
Command to run:
python parse_xml.py

3. task1.py
This program takes the vector space, number of latent semantics to identify and the algorithm used as input. It then identifies the latent semantics of the chosen vector space using the chosen algorithm and saves the latent semantics ordered by decreasing term-weights into an output file named as task1_<vector_space>_<algorithm>_latentsemantics.csv
Command to run:
python task1.py

4. task2.py
This program takes the vector space, number of latent semantics to identify, algorithm, user id, image id and location id as input. It then identifies the latent semantics of the chosen vector space using the chosen algorithm. It then projects all the vector spaces onto the latent semantics. Next, for a given user id, image id and location id, it identifies the top 5 matching users, images and locations and reports it.
Command to run:
python task2.py

5. task3.py
This program takes the chosen visual descriptor model, number of latent semantics to identify, algorithm and the image id as input. It creates the vector space by combining all images of the chosen model. It then identifies the latent semantics of the vector space and projects the data onto the new space. It then compares the given image with every image and computes the top 5 similar images. It then compares the image with every image of each location and calculates the average similarity which is taken as the similarity of the image with a particular location. It then reports the top 5 similar locations based on the calculated similarity.
Command to run:
python task3.py

6. task4.py
This program takes the chosen visual descriptor model, number of latent semantics to identify, algorithm and the location id as input. It then identifies the latent semantics.
Command to run:
python task4.py

7. task5.py
This program takes the location id, algorithm and number of latent semantics to identify as input. It then identifies the latent semantics based on the chosen location and projects the other locations onto the new latent semantics. It then computes the similarities of locations in the new space and reports the top 5 similar locations.
Command to run:
python task5.py

8. task6.py
This program reduces the dimensionality of a location-location similarity matrix created using the textual and visual features of all the locations and displays the latent semantics in terms of location weight pairs. The reduced dimension is taken as input from the user (k).
Command to run:
python task6.py

9. task7.py
This program takes the rank(k) as input. It then creates the 3 mode tensor of users x images x locations where each element is the common terms between a user, image and location. It then performs a rank k CP decomposition to reduce the dimensions. It then performs k means on each of the factor matrices to get k groups of clusers of users, images and locations and reports them in a file task7_groups.txt
Command to run:
python task7.py


Authors
-------
1. Abhyudaya Srinet
2. Adhiraj Tikku
3. Anjali Singh
4. Darshan Dagly
5. Laveena Sachdeva
6. Vatsal Sodha