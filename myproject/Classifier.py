import cv2
from cv2 import COLOR_BGR2GRAY
import numpy as np
import os

path = 'Image'
orb = cv2.ORB_create(nfeatures=1000)
### Import Images
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    print(cl)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList
def findID(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = [0, ]
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    finalVal = matchList.index(max(matchList))
    return finalVal

desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture(0)
while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    # img2 = cv2.cvtColor(img2, cv2, COLOR_BGR2GRAY)
    id = findID(img2, desList)
    print()
    cv2.putText(img2, classNames[id-1], (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    cv2.imshow('img2', img2)
    cv2.waitKey(1)