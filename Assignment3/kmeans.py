import pandas as pd
import math
import random
import time

EUCLIDEAN = 1
COSINE = 2
GEN_JACCARD = 3

# Loads a CSV files into a list of tuples.
# Ignores the first row of the file (header).
# Numeric attributes are converted to floats, nominal attributes
# are represented with strings.
# Parameters:
#   fileName: name of the CSV file to be read
# Returns: a list of tuples
def loadCSV(dataFile, labelFile):
    dataset = []
    with open(dataFile) as dataFile, open(labelFile) as labelFile:
        for dataLine, labelLine in zip(dataFile, labelFile):
            completeLine = dataLine.strip() + ',' + labelLine.strip()
            instance = lineToTuple(completeLine)
            dataset.append(instance)
    return dataset

# Converts a comma separated string into a tuple
# Parameters
#   line: a string
# Returns: a tuple
def lineToTuple(line):
    # remove leading/trailing witespace and newlines
    #cleanLine = line.strip()
    # separate the fields
    lineList = line.split(",")
    # convert strings into numbers
    stringsToNumbers(lineList)
    lineTuple = tuple(lineList)
    return lineTuple

# Destructively converts all the string elements representing numbers
# to floating point numbers.
# Parameters:
#   myList: a list of strings
# Returns None
def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
            myList[i] = float(myList[i])

# Checks if a given string can be safely converted into a positive float.
# Parameters:
#   s: the string to be checked
# Returns: True if the string represents a positive float, False otherwise
def isValidNumberString(s):
  if len(s) == 0:
    return False
  if  len(s) > 1 and s[0] == "-":
      s = s[1:]
  for c in s:
    if c not in "0123456789.":
      return False
  return True

def calculateEuclideanDistance(point1, point2):
    sumOfSquares = 0
    for i in range(0, len(point1)-1):
        sumOfSquares += ((point1[i] - point2[i]) ** 2)
    return math.sqrt(sumOfSquares)

def calculateCosineDistance(point1, point2):
    dotProduct = 0
    sumOfSquaresPoint1 = 0
    sumOfSquaresPoint2 = 0
    for i in range(0, len(point1)-1):
        dotProduct += point1[i] * point2[i]
        sumOfSquaresPoint1 += point1[i] ** 2
        sumOfSquaresPoint2 += point2[i] ** 2
    return (1 - (dotProduct / math.sqrt(sumOfSquaresPoint1 * sumOfSquaresPoint2)))

def calculateGeneralizedJaccardDistance(point1, point2):
    sumOfMins = 0
    sumOfMaxs = 0
    for i in range(0, len(point1)-1):
        sumOfMins += min(point1[i], point2[i])
        sumOfMaxs += max(point1[i], point2[i])
    return 1 - (sumOfMins/sumOfMaxs)

def distance(point1, point2, type):
    if type == EUCLIDEAN:
        return calculateEuclideanDistance(point1, point2)
    if type == COSINE:
        return calculateCosineDistance(point1, point2)
    if type == GEN_JACCARD:
        return calculateGeneralizedJaccardDistance(point1, point2)

def meanInstance(instanceList):
    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    means = [0] * (numAttributes-1)
    for instance in instanceList:
        for i in range(0, numAttributes-1):
            means[i] += instance[i]
    for i in range(0, numAttributes-1):
        means[i] /= float(numInstances)
    return tuple(means)

def assign(instance, centroids, distanceType):
    minDistance = distance(instance, centroids[0], distanceType)
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i], distanceType)
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex

def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def assignAll(instances, centroids, type):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids, type)
        clusters[clusterIndex].append(instance)
    return clusters

def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        centroid = meanInstance(clusters[i])
        centroids.append(centroid)
    return centroids

def kmeans(instances, k, initCentroids=None, distanceType=EUCLIDEAN):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
    else:
        centroids = initCentroids
    prevCentroids = []
    iterations = 0
    while (centroids != prevCentroids):
        iterations += 1
        clusters = assignAll(instances, centroids, distanceType)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)
        withinss = computeWithinss(clusters, centroids, distanceType)
        sse = computeSSE(clusters, centroids, distanceType)
    predictionAccuracy = calculatePredictionAccuracy(clusters)
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["sse"] = sse
    result["predictionAccuracy"] = predictionAccuracy
    result["iterations"] = iterations
    return result

def kmeansMultipleStoppingCond(instances, k, initCentroids=None, distanceType=EUCLIDEAN, maxIterations=500):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids
        random.seed(42)
        centroids = random.sample(instances, k)
    else:
        centroids = initCentroids
    prevCentroids = []
    prevClusters = []
    prevWithinss = float("inf")
    previousSSE = float("inf")
    sse = 0
    currentIterations = 0
    while (centroids != prevCentroids or currentIterations > maxIterations):
        if currentIterations > 0:
            prevClusters = clusters
            prevCentroids = centroids
            prevWithinss = withinss
            previousSSE = sse
        clusters = assignAll(instances, centroids, distanceType)
        centroids = computeCentroids(clusters)
        withinss = computeWithinss(clusters, centroids, distanceType)
        sse = computeSSE(clusters, centroids, distanceType)
        if (sse > previousSSE):
            sse = previousSSE
            clusters = prevClusters
            withinss = prevWithinss
            centroids = prevCentroids
            break
        currentIterations += 1
    predictionAccuracy = calculatePredictionAccuracy(clusters)
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["sse"] = sse
    result["predictionAccuracy"] = predictionAccuracy
    result["iterations"] = currentIterations
    return result

def calculatePredictionAccuracy(clusters):
    totalPredictionCount = 0
    totalAccuratePredictionsCount = 0
    for cluster in clusters:
        labelCount = [0] * 10
        for i in range(len(cluster)):
            instance = cluster[i]
            groundTruthLabel = int(instance[-1])
            labelCount[groundTruthLabel] = labelCount[groundTruthLabel] + 1
        maxLabelCount = 0
        for i in range(0, len(labelCount)):
            totalPredictionCount += labelCount[i]
            if labelCount[i] > maxLabelCount:
                maxLabelCount = i
        totalAccuratePredictionsCount += labelCount[maxLabelCount]
    return totalAccuratePredictionsCount / totalPredictionCount
            

def computeSSE(clusters, centroids, distanceType):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += distance(centroid, instance, distanceType) ** 2
    return result

def computeWithinss(clusters, centroids, distanceType):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += distance(centroid, instance, distanceType)
    return result

# Repeats k-means clustering n times, and returns the clustering
# with the smallest withinss
def repeatedKMeans(instances, k, n):
    bestClustering = {}
    bestClustering["withinss"] = float("inf")
    for i in range(1, n+1):
        #print("k-means trial %d," % i ,
        trialClustering = kmeans(instances, k)
        #print "withinss: %.1f" % trialClustering["withinss"]
        if trialClustering["withinss"] < bestClustering["withinss"]:
            bestClustering = trialClustering
            minWithinssTrial = i
    #print "Trial with minimum withinss:", minWithinssTrial
    return bestClustering

    

dataset = loadCSV("/Users/jessicahalterman/Documents/MachineLearning/Assignment3/data.csv", "/Users/jessicahalterman/Documents/MachineLearning/Assignment3/label.csv")

#initialize centroids consistently across all calls
random.seed(time.time())
centroids = random.sample(dataset, 10)

print("basic k-means with Eucliean distance")
clustering = kmeans(dataset, 10, centroids, EUCLIDEAN)
print(clustering["sse"])
print(clustering["withinss"])
print(clustering["predictionAccuracy"])
print(clustering["iterations"])

print("enhanced k-means with Eucliean distance")
clustering = kmeansMultipleStoppingCond(dataset, 10, centroids, EUCLIDEAN)
print(clustering["sse"])
print(clustering["withinss"])
print(clustering["predictionAccuracy"])
print(clustering["iterations"])

#showDataset2D(dataset)
#clustering = kmeans(dataset, 10, None, GEN_JACCARD)
print("basic k-means with Cosine distance")
clustering = kmeans(dataset, 10, centroids, COSINE)
print(clustering["sse"])
print(clustering["withinss"])
print(clustering["predictionAccuracy"])
print(clustering["iterations"])

print("enhanced k-means with Cosine distance")
clustering = kmeansMultipleStoppingCond(dataset, 10, centroids, COSINE)
print(clustering["sse"])
print(clustering["withinss"])
print(clustering["predictionAccuracy"])
print(clustering["iterations"])

#clustering = kmeans(dataset, 10, None, GEN_JACCARD)
print("basic k-means with Generalized Jaccard distance")
clustering = kmeans(dataset, 10, centroids, GEN_JACCARD)
print(clustering["sse"])
print(clustering["withinss"])
print(clustering["predictionAccuracy"])
print(clustering["iterations"])

print("enhanced k-means with Generalized Jaccard distance")
clustering = kmeansMultipleStoppingCond(dataset, 10, centroids, GEN_JACCARD)
print(clustering["sse"])
print(clustering["withinss"])
print(clustering["predictionAccuracy"])
print(clustering["iterations"])
