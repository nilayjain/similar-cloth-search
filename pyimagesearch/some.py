import cv2
import numpy as np
from sklearn.cluster import KMeans

#converting image to Lab Color space and then applying K-means

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, bin_edges) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    print bin_edges
    # return the histogram
    return hist

image = cv2.imread('some.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image = image.reshape((image.shape[0] * image.shape[1], 3))
clt = KMeans(n_clusters=9)
clt.fit(image)
#then i am calculating weights of these clusters using this   centroid_histogram() function 

hist = centroid_histogram(clt)
centers = clt.cluster_centers_
#now hist contains the weights of clusters and centers contain the cluster points in Lab color space.