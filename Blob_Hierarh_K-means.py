# USAGE
# python Blob_Hierarh_K-means.py -v 2018-10-08@13-38-29 



import cv2
import numpy as np
import argparse
import time
from scipy.cluster.hierarchy import linkage, fcluster #dendrogram,
from matplotlib import pyplot as plt

# Compile time arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Video link .avi, extension")
args = vars(ap.parse_args())

start = time.time()

AR = []
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.filterByCircularity = True
params.filterByInertia = True
params.filterByColor = True

params.minThreshold = 10
params.maxThreshold = 200
params.minArea = 50
params.maxArea = 200
params.minCircularity = .3
params.minInertiaRatio = .5
params.minDistBetweenBlobs = 2

def blob_col(col):
	params.blobColor = col

video = cv2.VideoCapture(str(args["video"])+'.avi')

ramp_frames = 30
for i in xrange(ramp_frames):
	ret, frame = video.read()	#FIRST REAL FRAME

min_cnt = 2500000			#Zero value in difference array

while 1:
	ret, frame1 = video.read()
	if ret == False:
		cv2.imwrite(str(args["video"])+"_stable.png", st_frame)
		break
	img = frame-frame1
	if np.count_nonzero(img) < min_cnt:
		st_frame = frame1
		min_cnt = np.count_nonzero(img)
		continue
	frame = frame1

img = cv2.imread(str(args["video"])+"_stable.png")

#BLACK Blobs
blob_col(0)
detector1 = cv2.SimpleBlobDetector_create(params)
keypoints1 = detector1.detect(img)
img_keypt = cv2.drawKeypoints(img, keypoints1, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#WHITE Blobs
blob_col(255)
detector2 = cv2.SimpleBlobDetector_create(params)
keypoints2 = detector2.detect(img)
img_keypt = cv2.drawKeypoints(img, keypoints2, np.array([]), (255, 0, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print "BLACK POINTS: ", len(keypoints1)
print "WHITE POINTS: ", len(keypoints2)
print "TOTAL SCORE : ", len(keypoints1)+len(keypoints2)

for i in keypoints1:
	(x,y) = i.pt
	AR.append([x,y])
	img_keypt = cv2.circle(img_keypt, (int(x),int(y)), 5, (0,255,255), -1 )

for i in keypoints2:
	(x,y) = i.pt
	AR.append([x,y])
	img_keypt = cv2.circle(img_keypt, (int(x),int(y)), 5, (0,0,0), -1 )

Z = np.vstack(AR)
Z = np.double(Z)

linked = linkage(Z, 'ward')
max_d = 100
clusters = fcluster(linked, max_d, criterion='distance' )
k = np.amax(clusters)
print "NO OF DICE  : ", k

Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,k ,None,criteria,5,cv2.KMEANS_PP_CENTERS)
i = 0
while(i<k):
	print "Dice: ",i
	A = Z[label.ravel()==i]
	x,y = A[0]
	cv2.putText(img_keypt,"Score: "+str(len(A)) , (int(x+50),int(y)),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
	i = i+1

print "Execution time is ", time.time()-start
img_keypt = cv2.resize(img_keypt, (0,0), fx= 0.5, fy= 0.5)
cv2.imshow("READ DATA", img_keypt)
cv2.imwrite(str(args["video"])+"_stablized.png", img_keypt)
cv2.waitKey()

#cophenet
