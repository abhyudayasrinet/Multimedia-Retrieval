Phase 1
-----------
In this phase we experiment with retrieval using textual and visual features for a set of users, locations and images. More information of the dataset can be found [here](http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/Div150Cred_readme.txt). We map the data into a vector space and implement various similarity measures to fetch similar users/images/locations from the database.

The dataset can be downloaded from the `devset	` folder [here](http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/)

More information on the tasks can be found in the task_descriptions.pdf file.

Phase 2
--------

The goal of this phase of the project is to identify the latent semantics using different dimensionality reduction algorithms and use them to find similarity between objects in the reduced feature space.

The dataset consists of data about certain number of locations, their corresponding images and the users who uploaded the images. The description and tags of the images have been used to find the TF, DF, TF-IDF values for users, images and locations. Visual descriptors of all the images are also given based on 10 models - CN, CM, CN3x3, CM3x3, HOG, LBP, LBP3x3, GLRLM, GLRLM3x3, CSD grouped by locations.


Phase 3
--------

In this phase we experiment with graphical analysis, indexing, classification and clustering.

For this phase, we work only with the images of the dataset. We begin by representing the images in a graph with an image being connected to its k most similar images. We then implement confidence clustering and spectral clustering to find clusters in the graph. We implement PageRank and Personalized PageRank to search for similar images. We implement Locality Sensitive Hashing to build an index structure for retrieval of the images. Finally, we perform classification using k nearest neighbours and Personalized PageRank.


Authors
-------
1. Abhyudaya Srinet
2. Adhiraj Tikku
3. Anjali Singh
4. Darshan Dagly
5. Laveena Sachdeva
6. Vatsal Sodha