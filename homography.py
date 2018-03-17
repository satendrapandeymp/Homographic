import numpy as np
import cv2, os, sys

# Load the images in gray scale

train = "images/Sample.jpg"
img2 = cv2.imread(train, 0)
factor = .3
img2 = cv2.resize(img2, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

def run(img1):
	img1 = cv2.resize(img1, None, fx=.6, fy=.6, interpolation=cv2.INTER_CUBIC)
	# Detect the SIFT key points and compute the descriptors for the two images
	sift = cv2.xfeatures2d.SIFT_create()
	keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
	keyPoints2, descriptors2 = sift.detectAndCompute(img2, None)

	# Create brute-force matcher object
	bf = cv2.BFMatcher()

	# Match the descriptors
	matches = bf.knnMatch(descriptors1, descriptors2, k=2)

	# Select the good matches using the ratio test
	goodMatches = []

	for m, n in matches:
	    if m.distance < 0.9 * n.distance:
		goodMatches.append(m)

	# Apply the homography transformation if we have enough good matches
	MIN_MATCH_COUNT = 5

	if len(goodMatches) > MIN_MATCH_COUNT:
	    # Get the good key points positions
	    sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
	    destinationPoints = np.float32([ keyPoints2[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)

	    # Obtain the homography matrix
	    M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)

	    # PLOTING Homographic image
	    im_out = cv2.warpPerspective(img1, M, (img2.shape[1],img2.shape[0]))

	else:
	    print("Not enough matches are found - %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))
	    matchesMask = None
	    im_out = img2

	# Display the results
	return im_out
