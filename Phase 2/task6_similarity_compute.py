import numpy
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

def sim_images(disMatrix):

	indexes = []

	#get the max value of the matrix
	maximum = numpy.amax(disMatrix)

	for i in range(0,3):

		#get the index of minimum value from the distance matrix
		minIndex = numpy.unravel_index(numpy.argmin(disMatrix, axis=None), disMatrix.shape)
		indexes.append([minIndex[0], minIndex[1]])

		#set the minimum value as the maximum so in next iteration we get the second minimum value of the matrix
		disMatrix[minIndex[0], minIndex[1]] = maximum

	return indexes



def CM(imgsvec1, imgsvec2, imgpairs=True):

	#calculate the pair wise manhattan distance between two vector matrices
	allDis = manhattan_distances(imgsvec1, imgsvec2)

	#average of all the distances between one vector of first matrix to all the vectors of second matrix, for all vectors of first matrix
	imgDis = numpy.mean(allDis, axis=1)

	#average of all the distances from previous step to get the final combined distance between all images
	finalDis = numpy.mean(imgDis, axis=0)

	#if return most similar image pairs is True, also return the most similar images
	if imgpairs:

		return finalDis, sim_images(allDis)

	return finalDis


def CM3x3(imgsvec1, imgsvec2, imgpairs=True):

	#number of features of the CM model
	features = 9

	start = 0

	#initialize an empty matrix 
	sumVec = numpy.zeros((imgsvec1.shape[0], imgsvec2.shape[0]))
	
	#loop over the 9 3x3 blocks
	for i in range(1,10):

		end = i*features

		#get the vector of this block
		img1block = imgsvec1[:,start:end]
		img2block = imgsvec2[:,start:end]

		
		#distance b/w block of image 1 and image 2
		blockDis = manhattan_distances(img1block, img2block)
		
		sumVec = sumVec + blockDis

		#start index of next block
		start = end

	#average of distance between all 9 blocks of two images
	sumVec = sumVec/9

	#average of all the distances between one vector of first matrix to all the vectors of second matrix, for all vectors of first matrix
	imgDis = numpy.mean(sumVec, axis=1)

	#average of all the distances from previous step
	finalDis = numpy.mean(imgDis, axis=0)

	#if return most similar image pairs is True
	if imgpairs:

		return finalDis, sim_images(sumVec)

	return finalDis


def CN(imgsvec1,imgsvec2, imgpairs=True):

	#calculate the pair wise euclidean distance between two vector matrices
	allDis = euclidean_distances(imgsvec1, imgsvec2)

	#average of all the distances between one vector of first matrix to all the vectors of second matrix, for all vectors of first matrix
	imgDis = numpy.mean(allDis, axis=1)

	#average of all the distances from previous step to get the final combined distance
	finalDis = numpy.mean(imgDis, axis=0)

	if imgpairs:

		return finalDis, sim_images(allDis)

	return finalDis


def CN3x3(imgsvec1,imgsvec2, imgpairs=True):

	#number of features of the CN model
	features = 11

	start = 0

	#initialize a matrix to store the distances of two images
	sumVec = numpy.zeros((imgsvec1.shape[0], imgsvec2.shape[0]))

	for i in range(1,10):

		end = i*features

		#distance b/w block of image 1 and image 2
		img1block = imgsvec1[:,start:end]
		img2block = imgsvec2[:,start:end]

		#distance b/w block of image 1 and image 2
		blockDis = euclidean_distances(img1block, img2block)

		sumVec = sumVec + blockDis

		#start index of the next image block
		start = end


	#average of distance between all 9 blocks of two images
	sumVec = sumVec/9

	#average of all the distances between one vector of first matrix to all the vectors of second matrix, for all vectors of first matrix
	imgDis = numpy.mean(sumVec, axis=1)

	#average of all the distances from previous step
	finalDis = numpy.mean(imgDis, axis=0)

	if imgpairs:

		return finalDis, sim_images(sumVec)

	return finalDis


def CSD(imgsvec1, imgsvec2, imgpairs=True):

	#calculate the pair wise euclidean distance between two vector matrices
	allDis = euclidean_distances(imgsvec1, imgsvec2)

	#average of all the distances between one vector of first matrix to all the vectors of second matrix, for all vectors of first matrix
	imgDis = numpy.mean(allDis, axis=1)

	#average of all the distances from previous step to get the final combined distance
	finalDis = numpy.mean(imgDis, axis=0)

	if imgpairs:

		return finalDis, sim_images(allDis)

	return finalDis


def GLRLM(imgsvec1, imgsvec2, imgpairs=True):

	#calculate the pair wise euclidean distance between two vector matrices
	allDis = euclidean_distances(imgsvec1, imgsvec2)

	#average of all the distances between one vector of first matrix to all the vectors of second matrix, for all vectors of first matrix
	imgDis = numpy.mean(allDis, axis=1)

	#average of all the distances from previous step to get the final combined distance
	finalDis = numpy.mean(imgDis, axis=0)

	if imgpairs:

		return finalDis, sim_images(allDis)

	return finalDis


def GLRLM3x3(imgsvec1, imgsvec2, imgpairs=True):

	#number of features of the CN model
	features = 44

	start = 0

	#initialize a matrix to store the distances of two images
	sumVec = numpy.zeros((imgsvec1.shape[0], imgsvec2.shape[0]))

	#loop over the 9 3x3 blocks
	for i in range(1,10):

		end = i*features

		#get the vector of this block
		img1block = imgsvec1[:,start:end]
		img2block = imgsvec2[:,start:end]

		#distance b/w block of image 1 and image 2
		blockDis = euclidean_distances(img1block, img2block)
		
		sumVec = sumVec + blockDis

		#start index of the next image block
		start = end


	#average of distance between all 9 blocks of two images
	sumVec = sumVec/9

	#average of all the distances between one vector of first matrix to all the vectors of second matrix, for all vectors of first matrix
	imgDis = numpy.mean(sumVec, axis=1)

	#average of all the distances from previous step to get the final combined distance
	finalDis = numpy.mean(imgDis, axis=0)

	if imgpairs:

		return finalDis, sim_images(sumVec)

	return finalDis


def HOG(imgsvec1, imgsvec2, imgpairs=True):

	#calculate the pair wise euclidean distance between two vector matrices
	allDis = euclidean_distances(imgsvec1, imgsvec2)

	#average of all the distances between one vector of first matrix to all the vectors of second matrix, for all vectors of first matrix
	imgDis = numpy.mean(allDis, axis=1)

	#average of all the distances from previous step to get the final combined distance
	finalDis = numpy.mean(imgDis, axis=0)

	if imgpairs:

		return finalDis, sim_images(allDis)

	return finalDis


def LBP(imgsvec1, imgsvec2, imgpairs=True):

	#calculate the pair wise euclidean distance between two vector matrices
	allDis = euclidean_distances(imgsvec1, imgsvec2)

	#average of all the distances between one vector of first matrix to all the vectors of second matrix, for all vectors of first matrix
	imgDis = numpy.mean(allDis, axis=1)

	#average of all the distances from previous step to get the final combined distance
	finalDis = numpy.mean(imgDis, axis=0)

	if imgpairs:

		return finalDis, sim_images(allDis)

	return finalDis


def LBP3x3(imgsvec1, imgsvec2, imgpairs=True):

	#number of features of this LBP model
	features = 16

	start = 0

	sumVec = numpy.zeros((imgsvec1.shape[0], imgsvec2.shape[0]))

	#loop over the 9 3x3 blocks
	for i in range(1,10):

		end = i*features

		#get the vector of this block
		img1block = imgsvec1[:,start:end]
		img2block = imgsvec2[:,start:end]

		#calculate euclidean distance for the blocks
		blockDis = euclidean_distances(img1block, img2block)
		
		sumVec = sumVec + blockDis

		#start index of the next image block
		start = end


	#average of distance between all 9 blocks of two images
	sumVec = sumVec/9

	#average of all the distances between one vector of first matrix to all the vectors of second matrix, for all vectors of first matrix
	imgDis = numpy.mean(sumVec, axis=1)

	#average of all the distances from previous step to get the final combined distance
	finalDis = numpy.mean(imgDis, axis=0)

	if imgpairs:

		return finalDis, sim_images(sumVec)

	return finalDis
