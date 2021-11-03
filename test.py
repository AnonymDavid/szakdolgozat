import numpy as np
import cv2
import imutils

img = cv2.imread("circuits/cnn/train/battery/1.jpg")


cv2.imshow('img', img)
cv2.imshow('imgr', imutils.rotate_bound(img, 15))

cv2.waitKey(0)