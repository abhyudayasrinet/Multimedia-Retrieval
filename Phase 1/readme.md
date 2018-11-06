Phase 1
-----------
In this phase we experiment with retrieval using textual and visual features for a set of users, locations and images. More information of the dataset can be found [here](http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/Div150Cred_readme.txt). We map the data into a vector space and implement various similarity measures to fetch similar users/images/locations from the database.

The dataset can be downloaded from the `devset	` folder [here](http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/)

More information on the tasks can be found in the task_descriptions.pdf file.

Requirements
-----------------

- python37
- numpy==1.15.1
- scipy==1.1.0

  

SETUP PROJECT
---------------------

1. Download and install all required libraries as listed above under "Requirements"
use pip to install using the following commands
pip install numpy==1.15.1
pip install scipy==1.1.0

2. unzip the solution file to folder inside dataset folder (folder containing folders descvis, desctxt, devset_topics.xml, poiNameCorrespondences.txt)
Structure must look like
dataset_folder
   - ../desctxt/desctxt/devset_textTermsPerUser.txt
   - ../desctxt/desctxt/devset_textTermsPerImage.txt
   - ../desctxt/desctxt/devset_textTermsPerPOI.txt
   - ../poiNameCorrespondences.txt
   - ../devset_topics.xml
   - ../descvis/*
   - ../Code/generate_sparse.py
   - ../Code/main.py
   - ../Code/parse_xml.py
   - 
3. Create folders images, locations, users inside the Code folder
structure of Code folder must look like this
Code/
-images
-locations
-users
-main.py
-generate_sparse.py
-parse_xml.py  

4. Run the generate_sparse.py file to generate the .npy and .npz files in the images, locations and users folder
 
5. Run the parse_xml.py file to generate location_internal_name_dict.npy and location_name_dict.npy files inside the locations folder

6. Run the main.py file and follow on screen instructions to execute each task