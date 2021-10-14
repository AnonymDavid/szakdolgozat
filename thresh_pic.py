import numpy as np
from cv2 import cv2
import glob

pictures = glob.glob("circuits/cnn/tests/6.jpg")

for pic in pictures:
    img = cv2.imread(pic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))

    cv2.imwrite(pic, thresh)