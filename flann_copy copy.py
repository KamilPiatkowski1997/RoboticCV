import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read images
img_1 = cv2.imread("INPUT/Data2/4.jpg")
img_2o = cv2.imread("INPUT/Data2/3.jpg")
# Resize
# img_1 = cv2.resize(img_1o,(0,0), None, 0.15, 0.15)
img_2 = cv2.resize(img_2o,(0,0), None, 0.15, 0.15)
# Gray
img1= cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img2= cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# # Read images
# img_1o = cv2.imread("INPUT/Data2/4.jpg")
# img_2o = cv2.imread("INPUT/Data2/3.jpg")
# # Resize
# width_resize=450
# height_resize=600
# dim=(width_resize,height_resize)
# img_2 = cv2.resize(img_2o, dim, interpolation= cv2.INTER_AREA)
# img_1 = img_1o
# cv2.imshow("showme", img_1)

# Gray
img1= cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img2= cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

#sift
sift = cv2.SIFT_create()

# Get keypoints and descriptors
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)


# feature matching using flann function
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

descriptors_1 = np.float32(descriptors_1)
descriptors_2 = np.float32(descriptors_2)

matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

draw_params = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)
# print("matches",matches)

good_matches = []
for m,n in matches:
    if m.distance < 0.32*n.distance:
        good_matches.append(m)
# print("goodmaches", good_matches)
# matches = sorted(matches, key=lambda x:x.distance)

# Visualize the results
img3 = cv2.drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, None, **draw_params)
cv2.imshow("matches", img3)

# find minimal number of good maches
if len (good_matches) >= 4:
    src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    Matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
    # print(Matrix)
    h,w = img2.shape
    pts = np.float32([ [0,0], [0,h-1], [w-1, h-1], [w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, Matrix)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255,3, cv2.LINE_AA)
    cv2.imshow("overlap", img2)
    # print(dst)

dst_pts = cv2.warpPerspective(img_1, Matrix, ((img_1.shape[1] + img_2.shape[1]), (img_1.shape[1] + img_2.shape[1])))
dst_pts[0:img_2.shape[0], 0:img_2.shape[1]] = img_2

# cut1= dst_pts[0:600, 0:655]
# cut2= dst_pts[45:600, 0:900]

cv2.imshow("result", dst_pts)
# cv2.imwrite("INPUT/Data2/Result.jpg", cut)
cv2.waitKey(0)
# plt.imshow(dst)
# plt.show()