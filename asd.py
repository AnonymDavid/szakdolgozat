import numpy as np
from cv2 import cv2
from dataclasses import dataclass
from typing import List, NamedTuple
import math
import sys
import os.path

# temporary imports
import time

from numpy.lib.function_base import diff

# ----- CONSTANTS -----
CNT_DELETE_PERCENT = 15
POINT_SIMILARITY_COMPARE_AREA_RADIUS = 50
LINE_COUNT_CHECK_FOR_ROTATION = 15
LINE_ANGLE_CHECK_THRESHOLD = 5 # degrees, both ways
# temp:
PICTURE_SCALE = 25

# ----- TYPES -----
class Point(NamedTuple):
    x: int
    y: int

@dataclass
class Line:
    p1: Point
    p2: Point

# ----- METHODS -----
def getLineLength(l: Line) -> float:
    """Get the length of a line."""
    return math.sqrt( ((l.p2.x-l.p1.x)**2 + (l.p2.y-l.p1.y)**2) )

def getLineAngle(l: Line) -> float:
    """Get the angle of line with the horizontal axis."""
    dx = l.p2.x - l.p1.x
    dy = l.p2.y - l.p1.y
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)
    if angle < 0:
        angle = 360 + angle
    return angle
# Angle:
#                       90
#             135                45
# 
#        180          Origin           0
# 
#            225                315
# 
#                      270

def getLinesIntersection(l1: Line, l2: Line) -> Point:
    """Get the intersecion of 2 lines (infinite lines)."""
    x1 = l1.p1.x
    x2 = l1.p2.x
    x3 = l2.p1.x
    x4 = l2.p2.x
    y1 = l1.p1.y
    y2 = l1.p2.y
    y3 = l2.p1.y
    y4 = l2.p2.y

    D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)

    return Point(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/D, ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/D)

def convertHoughToLineList(hl) -> List[Line]:
    """Converts the output of HoughLineP to Line array."""
    lines = []
    for l in hl:
        lines.append(convertHoughToLine(l))
    
    return lines

def convertHoughToLine(hl) -> Line:
    """Converts a HoughLinesP format line to Line."""
    return Line(Point(hl[0][0], hl[0][1]), Point(hl[0][2], hl[0][3]))

def isSameLine(l1: Line, l2: Line) -> bool:
    """Check if the given lines are the same based on threir endpoint position and a position threshold constant."""
    if (isSamePoint(l1.p1, l2.p1) and isSamePoint(l1.p2, l2.p2)) or (isSamePoint(l1.p1, l2.p2) and isSamePoint(l1.p2, l2.p1)):
        return True
    return False

def isSamePoint(p1: Point, p2: Point) -> bool:
    """Check if the given points are the same based on an area threshold constant."""
    if abs(p1.x - p2.x) < POINT_SIMILARITY_COMPARE_AREA_RADIUS and abs(p1.y - p2.y) < POINT_SIMILARITY_COMPARE_AREA_RADIUS:
        return True
    return False

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
    

# ----- MAIN -----

t1 = time.perf_counter()

if not len(sys.argv) == 2:
    exit("No parameter given!")

if not os.path.isfile(str(sys.argv[1])):
    exit("The file does not exist!")

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]

thresh = 255 - thresh


# removing texts
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contourSizes = []

max = 0
cntC = 0

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
    x, y, w, h = cv2.boundingRect(approx)

    contourSizes.append([cnt, w, h])

    if contourSizes[max][1]*contourSizes[max][2] < contourSizes[cntC][1]*contourSizes[cntC][2]:
        max = cntC

    cntC += 1


for i in range(cntC):
    if contourSizes[i][1] < (contourSizes[max][1] * CNT_DELETE_PERCENT / 100) and contourSizes[i][2] < (contourSizes[max][2] * CNT_DELETE_PERCENT / 100):
        cv2.drawContours(thresh, [contourSizes[i][0]], 0, (0), -1)




# Rotate thresh via longest lines avg angle
linesP = list(cv2.HoughLinesP(thresh, 1, np.pi/180, 100, None, 100, 0))
if linesP is not None:
    linesP.sort(key=lambda l: getLineLength(convertHoughToLine(l)), reverse=True )

    if len(linesP) <= 10:
        linesP = linesP[:LINE_COUNT_CHECK_FOR_ROTATION]
    else:
        linesP = convertHoughToLineList(linesP)

        rotatelines = []
        checkedLineCount = 0
        differentLineCount = 0
        while checkedLineCount < len(linesP) and differentLineCount < LINE_COUNT_CHECK_FOR_ROTATION:
            line = linesP[checkedLineCount]
            i = 0
            while i < differentLineCount and not isSameLine(line, rotatelines[i]):
                i += 1
            if (i >= differentLineCount):
                rotatelines.append(line)
                differentLineCount += 1
            checkedLineCount += 1

    avgAngleDiff = 0
    for line in rotatelines:
        angle = getLineAngle(line)
        angle = angle - (int)(angle / 90)*90
        if (angle > 45):
            angle = angle - 90
        avgAngleDiff += angle
    avgAngleDiff /= LINE_COUNT_CHECK_FOR_ROTATION

    thresh = rotateImage(thresh, avgAngleDiff)
    gray = rotateImage(gray, avgAngleDiff)
    img = rotateImage(img, avgAngleDiff)



linesP = list(cv2.HoughLinesP(thresh, 1, np.pi/180, 100, None, 100, 0))
if linesP is not None:
    linesP = convertHoughToLineList(linesP)

    horizontal = []
    vertical = []
    for line in linesP:
        line_angle = getLineAngle(line)
        if abs(line_angle - (round(line_angle / 180) * 180)) < LINE_ANGLE_CHECK_THRESHOLD:
            horizontal.append(line)
            cv2.line(img, (line.p1.x, line.p1.y), (line.p2.x, line.p2.y), (0,0,250), thickness=10)
        elif abs(line_angle - (round(line_angle / 90) * 90)) < LINE_ANGLE_CHECK_THRESHOLD:
            vertical.append(line)
            cv2.line(img, (line.p1.x, line.p1.y), (line.p2.x, line.p2.y), (250,0,0), thickness=10)

#   WARP AFFINE
#       find interest points
#           top-, bottom-, left-, right most lines -> calc (imaginary) intersections -> 4 points
#       warp based on the points

top_line = left_line = img.shape[0] + img.shape[1]
bot_line = right_line = -1


line1 = Line(Point(142, 500), Point(250, 1200))
line2 = Line(Point(250, 368), Point(895, 497))
cv2.line(img, (142, 500), (250, 1200), (0,250,0), thickness=10)
cv2.line(img, (250, 368), (895, 497), (0,250,0), thickness=10)

p = getLinesIntersection(line1, line2)

cv2.circle(img, p, 10, (255,0,255), -1)

















# TODO
# ???
# HoughLinesP
# subtract lines: drawContours: black, bigger width
# cut out components, feed it to the CNN
# 
# output
# 
# optional: before cnn: display only bounding boxes so live feed can see the recognition

# ============================== CNN ====================================
# model = keras.models.load_model("symbolsModel.h5")
# 
# prediction = model.predict(thresh)
# if np.argmax(prediction[0]) == 0:
#     print("inductor")
# else:
#     print("resistor")

img    = cv2.resize(img, (int(img.shape[1]*PICTURE_SCALE/100), int(img.shape[0]*PICTURE_SCALE/100)))
thresh = cv2.resize(thresh, (int(thresh.shape[1]*PICTURE_SCALE/100), int(thresh.shape[0]*PICTURE_SCALE/100)))

cv2.imshow("img", img)
# cv2.imshow("thresh", thresh)

t2 = time.perf_counter()

print(f"\nTime: {t2 - t1:0.4f} seconds")

cv2.waitKey(0)