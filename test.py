import numpy as np
import cv2
import sys
import os.path


def resizeImage(src, percent: int):
    return cv2.resize(src, (int(src.shape[1]*percent/100), int(src.shape[0]*percent/100)))

def putOnCanvas(image, imgPercent):
    canvas = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')

    image = resizeImage(image, imgPercent)

    offsetX = round((canvas.shape[1] - image.shape[1]) / 2)
    offsetY = round((canvas.shape[0] - image.shape[0]) / 2)

    x1 = offsetX
    x2 = offsetX + image.shape[1]
    y1 = offsetY
    y2 = offsetY + image.shape[0]

    canvas[y1:y2, x1:x2] = image

    return canvas


if not len(sys.argv) == 2:
    exit("No parameter given!")

if not os.path.isfile(str(sys.argv[1])):
    exit("The file does not exist!")

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(gray, 7)

thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]

thresh = 255 - thresh

thresh = putOnCanvas(thresh, 60)

cv2.imshow('thresh', resizeImage(thresh, 15))
cv2.waitKey(0)