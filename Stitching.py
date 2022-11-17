import cv2
import os

mainFolder = 'INPUT/Data2'
myFolders = os.listdir(mainFolder)
path = mainFolder
images = []
myList = os.listdir(path)
print(f'total no of images detected {len(myList)}')
for imgN in myList:
    curIMG = cv2.imread(f'{path}/{imgN}')
    curIMG = cv2.resize(curIMG,(0,0), None, 0.2, 0.2)
    images.append(curIMG)
    
stitcher = cv2.Stitcher.create()
(status,result) = stitcher.stitch(images)
if (status == cv2.Stitcher_OK):
    print("success")
    cut= result[25:700, 25:800]
    cv2.imshow("result",cut)
    cv2.imwrite("Stitching_Result.jpg", cut)
    cv2.waitKey(0)
else:
    print("fail")