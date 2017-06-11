# KMEANS IMPLEMENTATION

import numpy as np
import random
import math
import matplotlib.pyplot as plt
data_samples = [[0.1, 0.4, 0.2], [0.6, 0.1, 0.5,], [0.4, 1.6, 0.9], [1.2, 3.4, 6.5],
           [9.2, 9.6, 8.6]]
datalen = 5
data = []
centroid = []
k = 2
d = 3
BIG_NUMBER = math.pow(10, 10)

class DataPoint:
    def __init__(self, a):
        self.a = a
        
    def set_data(self, a):
        self.a = a
    
    def get_data(self):
        return self.a

    def set_cluster(self, clusterNumber):
        self.clusterNumber = clusterNumber
    
    def get_cluster(self):
        return self.clusterNumber

def initialize_datapoints():
    # DataPoint objects' x and y values are taken from the SAMPLE array.
    # The DataPoints associated with LOWEST_SAMPLE_POINT and HIGHEST_SAMPLE_POINT are initially
    # assigned to the clusters matching the LOWEST_SAMPLE_POINT and HIGHEST_SAMPLE_POINT centroids.
    for i in range(datalen):
        newPoint = DataPoint(data_samples[i])
        newPoint.set_cluster(None)
        data.append(newPoint)
    return

def printdatapoints():
    for i in range(datalen):
        print("data", data[i].a, "cluster", data[i].clusterNumber)
    return
        
# Initialize k number of centroids of dimension d
# Initialize with some random data point in the data
def centroidInit():
    randlist = random.sample(range(0,datalen), k)
    for i in range (0, k):
        centroid.append(data_samples[randlist[i]].copy())
#        print("Centroid", centroid[i])
    return

def calculate_centroids():
    totalA = [0] * d
    totalInCluster = 0
    prevCentroid = 0
    centroidMove = [1] * d
    import copy
    for j in range(k):
        prevCentroid = copy.copy(centroid[j])
        for i in range(datalen):
            if(data[i].get_cluster() == j):
                for z in range(d):
                    totalA[z] += data[i].a[z]
                totalInCluster += 1
        
        if(totalInCluster > 0):
            print("Centroids at:")
            for z in range(d):
                centroid[j][z] = totalA[z] / totalInCluster
                print(prevCentroid[z], centroid[j][z] )
                if(prevCentroid[z] == centroid[j][z]):
                    centroidMove[z] = 0
    if (sum(centroidMove) == 0):
        return 1
    else:
        return 0
# Repeat until centroids stay unchanged
# Find average of all data points in each cluster k and reinitialize centroid,

def EuclideanDis(dataPoint, centroidPoint):
    # Calculate Euclidean distance.
    disDimensions = 0
    for i in range(d):
#        print("CentroidPoint",centroidPoint[i])
#        print("dataPoint",dataPoint.a[i])
        disDimensions = disDimensions + math.pow((centroidPoint[i] - dataPoint.a[i]), 2)
    return math.sqrt(disDimensions)


# loop thru dataset find out the nearest cluster for each data point
def cluster():
    clusterShift = [1] * datalen
    for i in range(datalen):
        bestMinimum = BIG_NUMBER
        currentCluster = None
        for j in range(k):
            distance = EuclideanDis(data[i], centroid[j])
            print("Distance", distance, BIG_NUMBER)
            if(distance < bestMinimum):
                bestMinimum = distance
                currentCluster = j
        if(data[i].get_cluster() != currentCluster):
            data[i].set_cluster(currentCluster)
            clusterShift[i] = 1
    printdatapoints()
    if(sum(clusterShift) == 0):
        return 1
    else:
        return 0

def kmeans():
    clusterShift = 1
    centroidMove = 1
    cluster()
    while(clusterShift and centroidMove):
        centroidMove = calculate_centroids()
        clusterShift = cluster()
    return
initialize_datapoints()
centroidInit()
kmeans()

# Get array of dataset, get dimensions d, number of clusters k


