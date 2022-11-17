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
sift = cv2.SIFT_create()

# Get keypoints and descriptors
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
keypoints_3, descriptors_3 = sift.detectAndCompute(img3, None)


# feature matching using flann function
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

descriptors_1 = np.float32(descriptors_1)
descriptors_2 = np.float32(descriptors_2)
descriptors_3 = np.float32(descriptors_3)

matches_left = flann.knnMatch(descriptors_1, descriptors_2, k=2)
matches_right = flann.knnMatch(descriptors_2, descriptors_3, k=2)


draw_params_green = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)
draw_params_blue = dict(matchColor=(255,0,0), singlePointColor=None, flags=2)

# print("matches",matches)

good_matches_left = []
for m,n in matches_left:
    if m.distance < 0.5*n.distance:
        good_matches_left.append(m)
good_matches_right = []
for m,n in matches_right:
    if m.distance < 0.6*n.distance:
        good_matches_right.append(m)


# Visualize the results
# img_left = cv2.drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches_left, None, **draw_params_green)
img_right = cv2.drawMatches(img_2, keypoints_2, img_3, keypoints_3, good_matches_right, None, **draw_params_blue)
img_left = cv2.drawMatches(img_1, keypoints_1, img_right, keypoints_2, good_matches_left, None, **draw_params_green)
plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)), plt.title('Detector')
plt.show()



if len (good_matches_left) >= 4:
    # print(good_matches_left)
    src_pts_l = np.float32([ keypoints_1[m.queryIdx].pt for m in good_matches_left]).reshape(-1,1,2)
    dst_pts_l = np.float32([ keypoints_2[m.trainIdx].pt for m in good_matches_left]).reshape(-1,1,2)

    Matrix_L, mask_L = cv2.findHomography(src_pts_l, dst_pts_l, cv2.RANSAC, 5.0)
    print(Matrix_L)
    h,w = img2.shape
    pts_l = np.float32([ [0,0], [0,h-1], [w-1, h-1], [w-1,0] ]).reshape(-1,1,2)
    dst_l = cv2.perspectiveTransform(pts_l, Matrix_L)
    img2 = cv2.polylines(img2, [np.int32(dst_l)], True, 255,3, cv2.LINE_AA)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('Homography left')
    plt.show()

if len (good_matches_right) >= 4:
    # print(good_matches_right)
    src_pts_r = np.float32([ keypoints_2[m.queryIdx].pt for m in good_matches_right]).reshape(-1,1,2)
    dst_pts_r = np.float32([ keypoints_3[m.trainIdx].pt for m in good_matches_right]).reshape(-1,1,2)

    Matrix_R, mask_R = cv2.findHomography(src_pts_r, dst_pts_r, cv2.RANSAC, 5.0)
    print(Matrix_R)
    h,w = img2.shape
    pts_r = np.float32([ [0,0], [0,h-1], [w-1, h-1], [w-1,0] ]).reshape(-1,1,2)
    dst_r = cv2.perspectiveTransform(pts_r, Matrix_R)
    img3 = cv2.polylines(img3, [np.int32(dst_r)], True, 255,3, cv2.LINE_AA)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)), plt.title('Homography right')
    plt.show()

dst_pts_l = cv2.warpPerspective(img_1, Matrix_L, ((img_1.shape[1] + img_2.shape[1]), (img_1.shape[1] + img_2.shape[1])))
dst_pts_l[0:img_2.shape[0], 0:img_2.shape[1]] = img_2
cut_l= dst_pts_l[0:600, 0:655]
plt.imshow(cv2.cvtColor(cut_l, cv2.COLOR_BGR2RGB)), plt.title('Result Left')
plt.show()

dst_pts_r = cv2.warpPerspective(cut_l, Matrix_R, ((cut_l.shape[1] + img_3.shape[1]), (cut_l.shape[1] + img_3.shape[1])))
dst_pts_r[0:img_3.shape[0], 0:img_3.shape[1]] = img_3
cut_r= dst_pts_r[20:600, 0:990]
plt.imshow(cv2.cvtColor(cut_r, cv2.COLOR_BGR2RGB)), plt.title('Entire Result')
cv2.imwrite("3SHIFT_Result.jpg", cut_r)
plt.show()