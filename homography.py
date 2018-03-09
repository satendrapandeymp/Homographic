import numpy as np
import cv2, os

# Load the images in gray scale

train = "images/" + raw_input("Name of Train file : ")
test = "images/" +  raw_input("Name of Test file : ")

img1 = cv2.imread(test, 0)
img2 = cv2.imread(train, 0)

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
    if m.distance < 0.8 * n.distance:
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

# Display the results
cv2.imwrite("images/test.png", im_out)

os.system("python scan.py --image images/test.png")
cv2.imshow('Homography', img2)
cv2.imshow('Test', im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
Put this in If section if you want.
# Apply the perspective transformation to the source image corners
h, w = img1.shape
corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
transformedCorners = cv2.perspectiveTransform(corners, M)

# Draw a polygon on the second image joining the transformed corners
img2 = cv2.polylines(img2, [np.int32(transformedCorners)], True, (255, 255, 255), 2, cv2.LINE_AA)

# To plot lines joining features
matchesMask = mask.ravel().tolist()
drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
result = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, goodMatches, None, **drawParameters)
'''
