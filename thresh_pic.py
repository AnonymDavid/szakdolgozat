import numpy as np
from cv2 import cv2

start = 5
end = 5
while start <= end:
    img = cv2.imread("circuits/tests/"+str(start)+".png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))

    cv2.imwrite("circuits/tests/"+str(start)+".jpg", thresh)

    start+=1