# KMEANS IMPLEMENTATION FOR A MULTIDIMENSTIONAL DATA
# INPUT THROUGH COMMAND LINE: 
    #INPUT FILE NAME WITH DATAPOINTS
    #NUMBER OF FEATURES OR DIMENSIONS
    #NUMBER OF CLUSTERS FOR THE KMEANS

import numpy as np
import random
import math

class DataCluster:
    def __init__(self, data):
        # datapoint with the multidimensional value as a list
        # and the cluster number it belongs to
        self.data = data
        self.clusterNum = None

class AllData:
    def __init__(self, data_cluster_list):
        # List of DataCLuster class with datpoint and their clusters
        self.data_cluster_list = data_cluster_list
        # Length of the dataset
        self.datalen = len(data_cluster_list)
        # Dimensions or Features of the dataset
        self.numFeatures = len(data_cluster_list[0].data)

# Initialize the class DataCLusters with the data points and the clusters they belong to
def initDataClusters(data_samples, numClusters):
    data_cluster_list = []
    for i in range(len(data_samples)):
        newPoint = DataCluster(data_samples[i])
        #newPoint.set_cluster(None)
        data_cluster_list.append(newPoint)
    allData = AllData(data_cluster_list)
    return allData

def printDataClusters(allData):
    for i in range(allData.datalen):
        print("data_clusters", allData.data_cluster_list[i].data, "cluster",
              allData.data_cluster_list[i].clusterNum)
    return
        
# Initialize k number of centroids of dimension d
# Initialize with some random data point in the data
def Initcentroid(centroids, allData, numClusters):
    randlist = random.sample(range(allData.datalen), numClusters)
    for i in range (numClusters):
        randno = randlist[i]
        centroids.append(allData.data_cluster_list[i].data.copy())
#        print("Centroid", centroid[i])
    return

# Recalculate the centroids with the mean of all the datapoint in each cluster
def repositionCentroids(centroids, numClusters, allData):
    totalA = [0] * allData.numFeatures
    totalInCluster = 0
    prevCentroid = 0
    centroidMove = [1] * allData.numFeatures
    import copy
    for j in range(numClusters):
        prevCentroid = copy.copy(centroids[j])
        for i in range(allData.datalen):
            if(allData.data_cluster_list[i].clusterNum == j):
                for z in range(allData.numFeatures):
                    totalA[z] += allData.data_cluster_list[i].data[z]
                totalInCluster += 1
        if(totalInCluster > 0):
            for z in range(allData.numFeatures):
                centroids[j][z] = totalA[z] / totalInCluster
                if(prevCentroid[z] == centroids[j][z]):
                    centroidMove[z] = 0
    # If the any centroid point is changed then return 1
    # in the calling function centroidMove will be 1 to indicate that clustering is not final
    # Otherwise return 0 to indicate centroidMove that clustering is done
    if (sum(centroidMove) == 0):
        return 1
    else:
        return 0

def EuclideanDis(data_cluster, numFeatures, centroid):
    # Calculate Euclidean distance of a datapoint from the centroid
    disDimensions = 0
    for i in range(numFeatures):
        disDimensions = disDimensions + math.pow((centroid[i] - data_cluster.data[i]), 2)
    return math.sqrt(disDimensions)

# loop thru dataset find out the nearest centroid for each data point
def clusterData(centroids, numClusters, allData):
    clusterShift = [1] * allData.datalen
    for i in range(allData.datalen):
        minDistance = None
        currentCluster = None
        for j in range(numClusters):
            distance = EuclideanDis(allData.data_cluster_list[i], 
                                    allData.numFeatures, centroids[j])
            if (minDistance == None):
                minDistance = distance
                currentCluster = j
            elif(distance < minDistance):
                minDistance = distance
                currentCluster = j
        # rearrange the clusters
        if(allData.data_cluster_list[i].clusterNum != currentCluster):
            allData.data_cluster_list[i].clusterNum = currentCluster
            clusterShift[i] = 1
    # If any data point has shifted the clusters then return 1
    # in the calling function clusterShift will be 1 to indicate that clustering is not final
    # Otherwise return 0 to indicate clusterShift that clustering is done
    if(sum(clusterShift) == 0):
        return 1
    else:
        return 0

def kmeans(data_samples, numClusters):
    centroids = []
    allData = initDataClusters(data_samples, 2)
    Initcentroid(centroids, allData, numClusters)
    clusterShift = 1
    centroidMove = 1
    clusterData(centroids, numClusters, allData)
    # Reposition the centroid until the coordinates of the centroids stay 
    # unchanged and the datapoints remain in the clusters
    while(clusterShift and centroidMove):
        centroidMove = repositionCentroids(centroids, numClusters, allData)
        clusterShift = clusterData(centroids, numClusters, allData)
    printDataClusters(allData)
    return centroids

if __name__== "__main__":
    
    data_samples = []
    data_feature = []
    i = 0
    
    # Take the string data points from the file name passed as first argument
    # Number of features of dimension of each datapoint as the second argument
    # Number of clusters needed as the third command line argument
    import sys
    inputfile = sys.argv[1]
    numFeatures = int(sys.argv[2])
    numClusters = int(sys.argv[3])

    # read the float numbers from the input file
    f = open(inputfile,'r')
    for n in f.read().split(' '):
        j = (i % numFeatures)
        if (j == 0 and len(data_feature) != 0):
        #create a multidimensional array data_samples 
        #based on number of features in dataset
            data_samples.append(data_feature)
            data_feature = []
            data_feature.append(float(n))
        else:
             data_feature.append(float(n))
        i = i + 1
    data_samples.append(data_feature)
    
    # Finally call Kmeans alogorithm to get the centroids
    centroids = kmeans(data_samples, numClusters)
    
    # Write the coordinates of the centroids in the output file clusters.txt
    outf = open('clusters.txt', 'w')
    for i in range(numClusters):
        print("Centroid", i+1, "at")
        for j in range(numFeatures):
            print(centroids[i][j])
            outf.write(str(centroids[i][j]) + ' ')
    f.close()
    outf.close()