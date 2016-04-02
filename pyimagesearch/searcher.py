# import the necessary packages
import numpy as np
import csv
import math
import ast
#from cv2 import *
import cv2
class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath

	def search(self, queryFeatures):
		limit = 10
		# initialize our dictionary of results
		results = {}
		#convert the queryFeatures to cvMat array
		k1 = cv2.cv.fromarray(queryFeatures)
		

		# open the index file for reading		
		with open(self.indexPath) as f:
			# initialize the CSV reader
			for line in f:
			#for row in reader:
				# parse out the image ID and features, then compute the
				# earth movers distance between the features in our index
				# and our query features
				#split line at ,
				#before , is the image id and after that is the features we indexed
				line.split(',',1)
				features = line.split(',',1)[1]
				features = ast.literal_eval(features)
				x = len(features)/4
				features = np.array(features, np.float32)
				features = features.reshape(x,4)
				k2 = cv2.cv.fromarray(features)
				count = 0
				d = cv2.cv.CalcEMD2(k1,k2,cv2.cv.CV_DIST_L2)
				results[line.split(',',1)[0]] = d
			# close the reader
			f.close()
		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		results = sorted([(v, k) for (k, v) in results.items()])

		# return our (limited) results
		return results[:limit]