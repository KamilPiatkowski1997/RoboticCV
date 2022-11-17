import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read images
img_1o = cv2.imread("INPUT/Data2/1.jpg")
img_2o = cv2.imread("INPUT/Data2/2.jpg")
img_3o = cv2.imread("INPUT/Data2/3.jpg")

# Resize
img_1 = cv2.resize(img_1o,(0,0), None, 0.15, 0.15)
img_2 = cv2.resize(img_2o,(0,0), None, 0.15, 0.15)
img_3 = cv2.resize(img_3o,(0,0), None, 0.15, 0.15)

# Gray
img1= cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img2= cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
img3= cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)

#sift
orb = cv2.ORB_create()
# Get keypoints and descriptors
keypoints_1, descriptors_1 = orb.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(img2, None)
keypoints_3, descriptors_3 = orb.detectAndCompute(img3, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_left = bf.match(descriptors_1,descriptors_2) # Match descriptors.
matches_left = sorted(matches_left, key=lambda x:x.distance) # Sort them in the order of their distance.

matches_right = bf.match(descriptors_2,descriptors_3) # Match descriptors.
matches_right = sorted(matches_right, key=lambda x:x.distance) # Sort them in the order of their distance.

img_right = cv2.drawMatches(img_2, keypoints_2, img_3, keypoints_3, matches_right[:10], None,matchColor=(255,0,0),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_left = cv2.drawMatches(img_1, keypoints_1, img_right, keypoints_2, matches_left[:17], None,matchColor=(0,255,0),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)), plt.title('Detector')
plt.show()

if len (matches_left) >= 4:
    src_pts_l = np.float32([ keypoints_1[m.queryIdx].pt for m in matches_left]).reshape(-1,1,2)
    dst_pts_l = np.float32([ keypoints_2[m.trainIdx].pt for m in matches_left]).reshape(-1,1,2)
    Matrix_L, mask_L = cv2.findHomography(src_pts_l, dst_pts_l, cv2.RANSAC, 5.0)
    h,w = img2.shape
    pts_l = np.float32([ [0,0], [0,h-1], [w-1, h-1], [w-1,0] ]).reshape(-1,1,2)
    dst_l = cv2.perspectiveTransform(pts_l, Matrix_L)
    img2 = cv2.polylines(img2, [np.int32(dst_l)], True, 255,3, cv2.LINE_AA)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('Homography left')
    plt.show()
else:
    print( "Not enough matches are found - {}/{}".format(len(matches_left), 4) )
    matchesMask = None

if len (matches_right) >= 4:
    src_pts_r = np.float32([ keypoints_2[m.queryIdx].pt for m in matches_right]).reshape(-1,1,2)
    dst_pts_r = np.float32([ keypoints_3[m.trainIdx].pt for m in matches_right]).reshape(-1,1,2)
    Matrix_R, mask_R = cv2.findHomography(src_pts_r, dst_pts_r, cv2.RANSAC, 5.5)
    h,w = img2.shape
    pts_r = np.float32([ [0,0], [0,h-1], [w-1, h-1], [w-1,0] ]).reshape(-1,1,2)
    dst_r = cv2.perspectiveTransform(pts_r, Matrix_R)
    img3 = cv2.polylines(img3, [np.int32(dst_r)], True, 255,3, cv2.LINE_AA)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)), plt.title('Homography right')
    plt.show()

else:
    print( "Not enough matches are found - {}/{}".format(len(matches_left), 4) )
    matchesMask = None

dst_pts_l = cv2.warpPerspective(img_1, Matrix_L, ((img_1.shape[1] + img_2.shape[1]), (img_1.shape[1] + img_2.shape[1])))
dst_pts_l[0:img_2.shape[0], 0:img_2.shape[1]] = img_2
cut_l= dst_pts_l[0:600, 0:655]
plt.imshow(cv2.cvtColor(cut_l, cv2.COLOR_BGR2RGB)), plt.title('Result Left')
plt.show()

dst_pts_r = cv2.warpPerspective(cut_l, Matrix_R, ((cut_l.shape[1] + img_3.shape[1]), (cut_l.shape[1] + img_3.shape[1])))
dst_pts_r[0:img_3.shape[0], 0:img_3.shape[1]] = img_3
cut_r= dst_pts_r[20:550, 0:790]
plt.imshow(cv2.cvtColor(cut_r, cv2.COLOR_BGR2RGB)), plt.title('Entire Result')
cv2.imwrite("3ORB_Result.jpg", cut_r)
plt.show()