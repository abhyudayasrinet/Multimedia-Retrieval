import csv
import xmltodict
import os
from task6_similarity_compute import *

#function to calculate location-location similarity values
def calc_sim():
    
    modelsFuncDict = {'CM': CM, 'CM3x3': CM3x3, 'CN': CN, 'CN3x3': CN3x3, 'CSD': CSD, 'GLRLM': GLRLM,
                      'GLRLM3x3': GLRLM3x3,
                      'HOG': HOG, 'LBP': LBP, 'LBP3x3': LBP3x3}

    print("\nCalculating...\n")

    locDict = {}

    with open('.\devset_topics.xml') as f:
        allTopics = xmltodict.parse(f.read())

    #create a mapping from location number to location name
    for i in range(0, len(allTopics['topics']['topic'])):
        locDict[allTopics['topics']['topic'][i]['number']] = allTopics['topics']['topic'][i]['title']

    #a list of all location names
    locs = list(locDict.values())

    final_scores = [] #numpy.empty((0,30))

    for floc in dict(sorted(locDict.items(), key=lambda x: x[1])):
        simsList = []

        inputLoc = locDict[floc]
        
        #loop over all the visual models
        for model in modelsFuncDict.keys():

            # get the data for the input location for this model
            inputf = open('.\descvis\img\\' + inputLoc + ' ' + model + '.csv', "r")
            inputr = csv.reader(inputf)

            inputvecs = []

            #get the vectors for all images of this location
            for row in inputr:
                inputvecs.append([float(v) for v in row[1:]])

            inputallVecs = numpy.stack(inputvecs)

            disDict = {}

            # loop over all the locations to compare
            for loc in dict(sorted(locDict.items(), key=lambda x: x[1])):

                f = open('.\descvis\img\\' + locDict[loc] + ' ' + model + '.csv', "r")
                r = csv.reader(f)

                vecs = []

                for row in r:
                    vecs.append([float(v) for v in row[1:]])

                allVecs = numpy.stack(vecs)

                #get the distance between images for this model
                sim = modelsFuncDict[model](inputallVecs, allVecs, False)

                disDict[locDict[loc]] = sim

            # scale the distance values by dividing all the distances by max distance
            maxvalue = max(disDict.values())

            finalDict = {k: float(v) / maxvalue for k, v in disDict.items()}

            simsList.append(list(finalDict.values()))

        # matrix where rows are models and columns are locations
        simMatrix = numpy.stack(simsList)
        

        # average distance of all models
        allLocScore = numpy.mean(simMatrix, axis=0)
        
        allLocScoreDict = {}

        
        for i, LocScore in enumerate(allLocScore):
            
            #subtract the distance from 1 to get the similarity score
            allLocScoreDict[locs[i]] = 1 - LocScore

        
        sorted_dict = dict(sorted(allLocScoreDict.items(), key=lambda x: x[0]))
        
        #final similarity values for each location , for this location
        values = list(sorted_dict.values())
        
        final_scores.append(values)
        

    #return the loc-loc similarity matrix
    return numpy.array(final_scores)
