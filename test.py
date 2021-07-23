import numpy as np
from cv2 import cv2


def resize(src, percent: int):
    return cv2.resize(src, (int(src.shape[1]*percent/100), int(src.shape[0]*percent/100)))


img = cv2.imread("circuits/pc1_r1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]

gray = cv2.GaussianBlur(thresh, (3,3), cv2.BORDER_DEFAULT)
canny = cv2.Canny(thresh, 50, 80)

cv2.imshow("canny", resize(canny, 25))

# linesP = list(cv2.HoughLinesP(thresh, 1, np.pi/180, 100, None, 100, 0))

cv2.imshow("out", resize(thresh, 25))

cv2.waitKey(0)
cv2.destroyAllWindows()