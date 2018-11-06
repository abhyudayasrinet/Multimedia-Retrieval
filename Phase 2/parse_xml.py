import xml.etree.ElementTree
import numpy as np

e = xml.etree.ElementTree.parse('./devset_topics.xml').getroot()
topic_id_dict = {}
location_internal_name_dict = {}
#parse the devset_topics.xml to build a dictionaries mapping name->id and id->title
for x in e.findall("topic"):
    location_title = x.find("title").text
    location_id = x.find("number").text
    topic_id_dict[location_title] = location_id
    location_internal_name_dict[int(location_id)] = location_title

#save the id->name dictionary
np.save('locations/location_internal_name_dict.npy', location_internal_name_dict)

location_dict = {}
f = open("./poiNameCorrespondences.txt", "r")
#build the mapping from id to readable name eg. 1->angel of the north
for line in f.readlines():
    words = line.strip().split('\t')
    title = ' '.join(words[0:-1])
    internal_title = words[-1]
    location_id = topic_id_dict[internal_title]
    location_dict[int(location_id)] = title

np.save('locations/location_name_dict.npy', location_dict)
