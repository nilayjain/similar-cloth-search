# USAGE
# python search.py --index index.csv --query queries/103100.png --result-path dataset

# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.searcher import Searcher
import argparse
import cv2
import numpy as np

e1 = cv2.getTickCount()
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
args = vars(ap.parse_args())

# initialize the image descriptor
cd = ColorDescriptor()

# load the query image and describe it
query = cv2.imread(args["query"])

features = cd.describe(query)


searcher = Searcher(args["index"])
results = searcher.search(features)

# display the query
cv2.imshow("Query", query)
e2 = cv2.getTickCount()
time = (e2-e1)/cv2.getTickFrequency()

# loop over the results
for (score, resultID) in results:
	# load the result image and display it
	result = cv2.imread(args["result_path"] + "/" + resultID)
	#print result
	cv2.imshow("Result", result)
	cv2.waitKey(0)
	print resultID
	print score

print "time taken is: " + str(time)
