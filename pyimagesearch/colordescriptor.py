import numpy as np
import cv2
import argparse
import math
import sys
import os
from sklearn.cluster import KMeans
#from matplotlib import pyplot as plt	

class ColorDescriptor:

	def isface(self, image):
		#convert to grayscale
		im = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		#face cascade
		cascade = './haarcascade_frontalface_alt2.xml'
		#equalize histogram and set parameters for detectMultiScale
		im = cv2.equalizeHist(im)
		side = math.sqrt(im.size)
		minlen = int(side / 20)
		maxlen = int(side / 2)
		flags = cv2.cv.CV_HAAR_DO_CANNY_PRUNING
		# frontal faces
		cc = cv2.CascadeClassifier(cascade)
		features = cc.detectMultiScale(im, 1.1, 4, flags, (minlen, minlen), (maxlen, maxlen))
		if len(features)==0:
			#if no face detected try the side face recognition
			cs = cv2.CascadeClassifier('./haarcascade_profileface.xml')
			features = cs.detectMultiScale(im,1.1,4,flags,(minlen,minlen), (maxlen,maxlen))
		if len(features)==0:
			#print('no face detected')
			return None
		im = image
		font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, cv2.cv.CV_AA)
		fontHeight = cv2.cv.GetTextSize("", font)[0][1] + 5
		#make a rectangle around the body based on the geometrical dimensions of face 
		#body is 7.5 heads long 
		for i in range(len(features)):	
			rect = features[i]
			xy1 = (rect[0], rect[1])
			xy2 = (rect[0] + rect[2], rect[1] + rect[3])
			x = rect[0]
			y = rect[1]
			w = rect[2]
			h = rect[3]
			rows,cols,channels = im.shape
			recx1=0
			recy1=0
			recx2=0
			recy2=0
			#cv2.rectangle(im, xy1, xy2, (255, 255, 255), 4)
			center = (xy1[0]+35, xy1[1]+35)
			im[np.where((im == im[center[1]][center[0]]).all(axis = 2))] = [0,0,0]
			recx1 = (int(x-w/2) if int(x-w/2)>0 else 0)
			recy1 = y+h
			recx2 = (x+int(3*w/2) if int(x+3*w/2)<cols else cols)
			recy2 = (y+int(4.5*h) if y+int(4.5*h)<rows else rows)#top ke liye 4.5 #saree 6.5
			#cv2.rectangle(im,(recx1,recy1),(recx2,recy2),(255,255,255),4)
			cropped = im[recy1:recy2, recx1:recx2]
			return cropped

	def centroid_histogram(self,clt):
		# grab the number of different clusters and create a histogram
		# based on the number of pixels assigned to each cluster
		numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
		(hist, _) = np.histogram(clt.labels_, bins = numLabels)
		# normalize the histogram, such that it sums to one
		hist = hist.astype("float")
		hist /= hist.sum()

		# return the histogram
		return hist

	def describe(self, image):
		try:
			#smooth the image to reduce effect of noise
			image = cv2.GaussianBlur(image,(5,5),0)
			img = None
			if image is None: return []
			img = self.isface(image)
			image2 = []
			ellipMask = []
			#if face is detected we have a rectangle otherwise
			#draw an ellipse at center of image
			if img is not None:
				image2 = img
			else:
				(h,w) = image.shape[:2]
				(cX, cY) = (int(w * 0.5), int(h * 0.5))
				(axesX, axesY) = (int(w * 0.25) / 2, int(h * 0.5) / 2)
				ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
				cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
				image2 = cv2.bitwise_and(image,image, mask = ellipMask)
			#don't resize image -- bad performance
			# can try increasing clusters to 7 or 8 but time penalty
			image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
			image2 = image2.reshape((image2.shape[0] * image2.shape[1], 3))
			#apply k means with 6 clusters
			clt = KMeans(n_clusters=6)
			clt.fit(image2)
			#extract the cluster weights
			hist = self.centroid_histogram(clt)
			centers = clt.cluster_centers_
			#join the cluster weights and centers and return
			k = np.c_[hist,centers]
			some = k.astype(np.float32)
			count = 0
			#delete clusters which have less than 0.1% representation
			for l in some:
				if l[0] < 0.001:
					some = np.delete(some,count,0)
					count = count-1
				count = count + 1
		except:
			print('some exception occured')
			some = None
		# return the feature vector
		return some
