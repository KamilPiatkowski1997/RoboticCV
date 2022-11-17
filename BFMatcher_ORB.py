import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

##################################################  Image manipulaiton

img1_o = cv.imread('INPUT/Data1/1.jpg')           # TrainImage 
img2_o = cv.imread('INPUT/Data1/2.jpg')           # QueryImage 

img1_r = cv.resize(img1_o,(0,0), None, 0.15, 0.15)  #resize
img2_r = cv.resize(img2_o,(0,0), None, 0.15, 0.15)

img1= cv.cvtColor(img1_r, cv.COLOR_BGR2GRAY)      #grayscale4Query
img2= cv.cvtColor(img2_r, cv.COLOR_BGR2GRAY)      #grayscale4Train

# img1 = cv.resize(img1,(0,0), None, 0.15, 0.15)  #resize
# img2 = cv.resize(img2,(0,0), None, 0.15, 0.15)  #resize

##################################################  Detectors

                                                    # Initiate ORB detector
orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

                                                    # create BFMatcher object with ORB detector (best?)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2) # Match descriptors.
matches = sorted(matches, key=lambda x:x.distance) # Sort them in the order of their distance.
# print(matches)
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:17], None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# plt.imshow(img3)
# plt.show()

plt.subplot(221), plt.imshow(cv.cvtColor(img2_o, cv.COLOR_BGR2RGB)), plt.title('Query Image ')
plt.subplot(222), plt.imshow(cv.cvtColor(img1_o, cv.COLOR_BGR2RGB)), plt.title('Train Image ')
plt.subplot(223), plt.imshow(img3), plt.title('Detector')

##################################################  Homography

if len(matches)>4:
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)

    matchesMask = mask.ravel().tolist()
    w,h = img1.shape #correct
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,H)
    # homography = cv.polylines(img2,[np.int32(dst)],True,(255, 0, 0), 1, cv.LINE_AA)
    # cv.imshow("homography",homography)
    # plt.subplot(224), plt.imshow(homography), plt.title('Homography')
else:
    print( "Not enough matches are found - {}/{}".format(len(matches), 4) )
    matchesMask = None

##################################################  Mask
empty = np.zeros_like(img2)
homography = cv.fillPoly(empty, [np.int32(dst)], (255, 0, 0), None)
masked_image = cv.bitwise_and(img2, homography)
# cv.imshow("masked_image",masked_image)
# plt.imshow(masked_image)
##################################################  Warp
width = img1.shape[1] + img2.shape[1]
height = img1.shape[1]

result = cv.warpPerspective(img1_r, H, (width,height))
result[0:img2_r.shape[0], 0:img2_r.shape[1]] = img2_r
cut= result[0:450, 0:]

##################################################  result
# cv.imshow("img1_r",cut)
# cv.imwrite("BFMatcher_ORB_Result.jpg", cut)
# cv.waitKey(0)
plt.subplot(224), plt.imshow(cv.cvtColor(cut, cv.COLOR_BGR2RGB)), plt.title('Result')
plt.show()