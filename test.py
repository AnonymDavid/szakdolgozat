import numpy as np
from cv2 import cv2
import math

img = cv2.imread("circuits/pc1_r1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]

ksize = 3
gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

cv2.imshow("Sobel/Scharr X", cv2.resize(gX, (int(gX.shape[1]*25/100), int(gX.shape[0]*25/100))))
cv2.imshow("Sobel/Scharr Y", cv2.resize(gY, (int(gY.shape[1]*25/100), int(gY.shape[0]*25/100))))
cv2.imshow("Sobel/Scharr Combined", cv2.resize(combined, (int(combined.shape[1]*25/100), int(combined.shape[0]*25/100))))

cv2.waitKey(0)
cv2.destroyAllWindows()