
from numpy import *
import random as r

## Code courtesy of Stephen Marsland, from his book "Machine Learning, An Algorithmic Perspective"
## Modified for use with thesis

def loadDataSet(fileName):      # function to parse delimited floats
    dataMat = []                # dataMat is a list
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine))      # map all elements to float()
        dataMat.append(fltLine)
    return dataMat              # returns list of lists of data points

def distEclud(vecA, vecB):      # returns Euclidean distance between points
    vecA = asarray(vecA)
    vecB = asarray(vecB)
    return sqrt(sum(power(vecA - vecB, 2)))   
    
def normalize(dataSet):
    n = len(dataSet[0])
    for z in range(n):
      attrs = [point[z] for point in dataSet]
      amean = mean(attrs)
      astd = std(attrs)
      if astd == 0:
        astd = 1
      for x in range(len(dataSet)):
        dataSet[x][z] = (dataSet[x][z] - amean) / astd
      
    return dataSet
    
def furthestPoint(dataSet,points):
    max_i = -1
    max_sep = 0.0
    for x in range(len(dataSet)):
      min_dist = inf
      for y in range(len(points)):
        dist = distEclud(dataSet[x],points[y])
        if dist < min_dist:
          min_dist = dist
          
      if min_dist > max_sep:
        max_i = x
        max_sep = min_dist
        
    return dataSet[max_i]

def randCent(dataSet, k, tactic=0):         # create k random initial cluster centers
    n = shape(dataSet)[1]         # n = number of dimensions of input examples
    centroids = mat(zeros((k,n))) # create k x n centroid matrix
    if tactic == 2:
      first = r.sample(range(len(dataSet)),1)
      cs = []
      cs.append(dataSet[first[0]])
      for i in range(k-1):
        next = furthestPoint(dataSet,cs)
        cs.append(next)
      for i in range(len(cs)):  
        centroids[i,:] = cs[i]
        
    elif tactic == 1:
      indices = r.sample(range(len(dataSet)),k)
      for i in range(len(indices)):
        centroids[i,:] = dataSet[indices[i]]
    else:
      for j in range(n):            # create random cluster centers (j = dimension)
                                    #   within bounds of each dimension
          minJ = min([point[j] for point in dataSet]) 
          maxJ = max([point[j] for point in dataSet])
          rangeJ = float(maxJ - minJ)
          centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))      
        
    return centroids    
    
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent,tactic=0):
    dataSet = asarray(dataSet)
    m = shape(dataSet)[0]               # m = number of data points
    clusterAssment = mat(zeros((m,2)))  # create matrix of data point assigns (col 0) 
                                        #   col 0: index of point's assigned cluster
                                        #   col 1: point's distance^2 to center
    centroids = createCent(dataSet, k, tactic)  # create k initial random centroids
    init_centroids = copy(centroids)
    #print("initial centroids=\n",around(centroids,2))
    clusterChanged = True               # run algorithm at least one iteration
    while clusterChanged:               # while cluster changes still occurring
        clusterChanged = False             # assume no changes occur (to terminate)
        # E step (assign data points to closest centroids)
        for i in range(m):                 # for each data point i
            minDist = inf; minIndex = -1
            for j in range(k):                       # for each centroid j
                distJI = distMeas(centroids[j,:],dataSet[i,:])  # dist between i, j
                if distJI < minDist:
                    minDist = distJI; minIndex = j       # assign i to closest centroid j
            if clusterAssment[i,0] != minIndex: clusterChanged = True  
            clusterAssment[i,:] = minIndex,minDist       # save i's cluster assnment, dist.
        #print("centroids =\n",around(centroids,3))
        # M step (recalculate centroid locations)
        for c in range(k):                # for each centroid c
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==c)[0]]
                                                       # get all points in cluster c
                                                       # self.A is np.asarray(self)
            centroids[c,:] = mean(ptsInClust, axis=0)  # assign centroid to mean 
    #print("final centroids =\n",around(centroids,2),"\n")
    #print "final assignments =\n",clusterAssment
    return centroids, clusterAssment, init_centroids


def p_results(results,k,tactic,normalize):
  print("************************************")
  # p_title(k,tactic,normalize)
  # p_centroids("Initial Centroids",results[2],False)
  # p_centroids("Final Centroids",results[0],True)
  clusterAsgn = results[1]
  print("By cluster: # points, mean dist to center")
  for c in range(k):
    cluster = nonzero(clusterAsgn[:,0].A==c)[0]
    mean_dist = mean([clusterAsgn[i,1] for i in cluster])
    print(c,":\t",len(cluster),"\t",around(mean_dist,2))
  print("Total mean dist from cluster assigned:", mean(clusterAsgn[:,1]))
  return mean(clusterAsgn[:,1])
      
def p_centroids(label,centroids,mat):
  print(label)
  if mat:
    for k in range(shape(centroids)[0]):
      print(k,":\t",end="")
      for x in range(shape(centroids)[1]):
        print(around(centroids[k,x],2),end="\t")
      print()
  else:
    k = 0
    for c in centroids:
      print(k,":\t",end="")
      k += 1
      for x in list(c):
        print(around(x,2),end="\t")
      print()
 
def p_title(k,tactic,normalize):
  title = str(k) + " clusters on data, "
  if tactic == 2:
    title = title + "Centroid Init: Furthest Points"
  elif tactic == 1:
    title = title + "Centroid Init: Random Data Points"
  else:
    title = title + "Centroid Init: Random Points in Space"
  if normalize:
    title = title + ", Data Normalized"
  print(title)

