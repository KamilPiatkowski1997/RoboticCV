import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import imutils

#Read number of images in the folder resize them and turn to grayscale
images_path = glob.glob('INPUT/Data2/*.jpg')
images = []
for image in images_path:
    img = cv.imread(image)
    img_r = cv.resize(img,(0,0), None, 0.15, 0.15)
    imgs= cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
    images.append(imgs)

#sift
sift = cv.SIFT_create()
keypoints = []
descriptors = []
for i in images:
    kp, ds = sift.detectAndCompute(i, None)
    ds= np.float32(ds)
    keypoints.append(kp)
    descriptors.append(ds)

