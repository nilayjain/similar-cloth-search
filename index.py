# USAGE
# python index.py --dataset dataset --index index.csv


# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
import argparse
import glob
import cv2
import numpy as np
import httplib
import urllib
import urllib2
import requests
from requests.auth import HTTPBasicAuth
import json
import ast
import os
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

# initialize the color descriptor
cd = ColorDescriptor()

# open the output index file for writing
output = open(args["index"], "w")
count = 0
for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(os.path.abspath(imagePath))
	# describe the image
	features = cd.describe(image)
	#ignore image if empty features
	if features ==[] or features is None: continue
	features = features.ravel().tolist()
	# write the features to file
	output.write("%s,%s\n" % (imageID, features))

# close the index file
output.close()
