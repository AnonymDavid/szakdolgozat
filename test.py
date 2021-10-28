import numpy as np
from cv2 import cv2
import math


def resize(src, percent: int):
    return cv2.resize(src, (int(src.shape[1]*percent/100), int(src.shape[0]*percent/100)))

def rotateImage(image, angle: int):
    """Rotates a given image by the given angle counterclockwise."""
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)

    rotated_image = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotated_image[0, 2] += ((bound_w / 2) - image_center[0])
    rotated_image[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_image = cv2.warpAffine(image, rotated_image, (bound_w, bound_h))

    return rotated_image


img = cv2.imread("circuits/cnn/train/transistor_pnp/21.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]



cv2.imshow("out", thresh)
cv2.imshow("filp", np.flip(thresh, 1))
cv2.imshow("rotate 90", rotateImage(thresh, 90))
cv2.imshow("rotate -90", rotateImage(thresh, -90))

cv2.waitKey(0)
cv2.destroyAllWindows()