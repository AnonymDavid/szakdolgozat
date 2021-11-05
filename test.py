import numpy as np
import cv2
from typing import List, NamedTuple
from enum import Enum
import math
import imutils
import sys
import os.path

import time

from numpy.core.fromnumeric import shape

# ----- CONSTANTS -----
POINT_SIMILARITY_COMPARE_AREA_RADIUS = 15
LINE_MIN_LENGTH = 90
LINE_COUNT_CHECK_FOR_ROTATION = 10
LINE_SEARCH_ANGLE_THRESHOLD = 5 # degrees, both ways
LINE_CHECK_SIMILARITY_THRESHOLD = 15
LINE_AGGREGATION_SIMILARITY_THRESHOLD = 25
COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH = 50
COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH = 350
COMPONENT_OTHER_ENDPOINT_SEARCH_MIN_LENGTH = 16
COMPONENT_MIN_BOX_SIZE = 200
COMPONENT_BOX_SIZE_OFFSET = 60
OUTPUT_POINT_SIMILARITY_COMPARE_AREA_RADIUS = 30
OUTPUT_SCALE = 3
PICTURE_SCALE = 20


# ----- TYPES -----
class Orientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1

class Point(NamedTuple):
    x: int
    y: int

class Line(NamedTuple):
    p1: Point
    p2: Point

class Endpoint(NamedTuple):
    point: Point
    orientation: Orientation


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
    l1_left, l1_right = [l1.p1, l1.p2] if l1.p1.x < l1.p2.x else [l1.p2, l1.p1]
    l2_left, l2_right = [l2.p1, l2.p2] if l2.p1.x < l2.p2.x else [l2.p2, l2.p1]
    
    l1_top, l1_bot = [l1.p1, l1.p2] if l1.p1.y < l1.p2.y else [l1.p2, l1.p1]
    l2_top, l2_bot = [l2.p1, l2.p2] if l2.p1.y < l2.p2.y else [l2.p2, l2.p1]
    
    if l1_left.x <= l2_right.x + LINE_CHECK_SIMILARITY_THRESHOLD and l1_right.x + LINE_CHECK_SIMILARITY_THRESHOLD >= l2_left.x and l1_top.y <= l2_bot.y + LINE_CHECK_SIMILARITY_THRESHOLD and l1_bot.y + LINE_CHECK_SIMILARITY_THRESHOLD >= l2_top.y:
        return True
    return False


def isSamePoint(p1: Point, p2: Point, threshold: int) -> bool:
    """Check if the given points are the same based on an area threshold constant."""
    if abs(p1.x - p2.x) < threshold and abs(p1.y - p2.y) < threshold:
        return True
    return False


def pointsCloseArray(p1: Point, points: List[Point]) -> int:
    """Check if the given point is close to another in the array."""
    for i in range(len(points)):
        if isSamePoint(p1, points[i], POINT_SIMILARITY_COMPARE_AREA_RADIUS):
            return i
    return -1

def resizeImage(src, percent: int):
    return cv2.resize(src, (int(src.shape[1]*percent/100), int(src.shape[0]*percent/100)))


def putOnCanvas(image, imgPercent):
    if len(image.shape) == 3:
        canvas = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype='uint8')
    else:
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


def followLine(lines, lineIdx, otherSideIdx, component_endpoints, checkedLines, outputLines, horizontalCount):
    if lineIdx in checkedLines:
        return
    
    line = lines[lineIdx]
            

    lineTemp = [[round(line.p1.x/OUTPUT_SCALE, -1), round(line.p1.y/OUTPUT_SCALE, -1)], [round(line.p2.x/OUTPUT_SCALE, -1), round(line.p2.y/OUTPUT_SCALE, -1)]]

    compEpCount = 0
    lineCompSide = -1
    while compEpCount < len(component_endpoints) and lineCompSide == -1:
        if isSamePoint(line.p1, component_endpoints[compEpCount], POINT_SIMILARITY_COMPARE_AREA_RADIUS):
            lineCompSide = 0
        elif isSamePoint(line.p2, component_endpoints[compEpCount], POINT_SIMILARITY_COMPARE_AREA_RADIUS):
            lineCompSide = 1
        else:
            compEpCount += 1
    
    if compEpCount < len(component_endpoints):
        lineTemp[lineCompSide] = [int(round(component_endpoints[compEpCount].x/OUTPUT_SCALE, -1)), int(round(component_endpoints[compEpCount].y/OUTPUT_SCALE, -1))]
    
    if lineIdx < horizontalCount:
        closestInterestDiff = 1000
        for olineC in range(len(outputLines)):
            interestDiff = abs(outputLines[olineC].p1.x - lineTemp[otherSideIdx][0])
            if interestDiff < closestInterestDiff:
                closestInterestDiff = interestDiff
                closestInterest = outputLines[olineC].p1.x
            
            interestDiff = abs(outputLines[olineC].p2.x - lineTemp[otherSideIdx][0])
            if interestDiff < closestInterestDiff:
                closestInterestDiff = interestDiff
                closestInterest = outputLines[olineC].p2.x
        
        if closestInterestDiff < 10:
            lineTemp[otherSideIdx][0] = closestInterest
        
        if lineCompSide == -1:
            lineTemp[otherSideIdx][1] = lineTemp[1-otherSideIdx][1]
        else:
            lineTemp[1-lineCompSide][1] = lineTemp[lineCompSide][1]
    else:
        closestInterestDiff = 1000
        for olineC in range(len(outputLines)):
            interestDiff = abs(outputLines[olineC].p1.y - lineTemp[otherSideIdx][1])
            if interestDiff < closestInterestDiff:
                closestInterestDiff = interestDiff
                closestInterest = outputLines[olineC].p1.y
            
            interestDiff = abs(outputLines[olineC].p2.y - lineTemp[otherSideIdx][1])
            if interestDiff < closestInterestDiff:
                closestInterestDiff = interestDiff
                closestInterest = outputLines[olineC].p2.y
        
        if closestInterestDiff < 10:
            lineTemp[otherSideIdx][1] = closestInterest
        
        if lineCompSide == -1:
            lineTemp[otherSideIdx][0] = lineTemp[1-otherSideIdx][0]
        else:
            lineTemp[1-lineCompSide][0] = lineTemp[lineCompSide][0]

    outputLines.append(Line(Point(round(lineTemp[0][0]), round(lineTemp[0][1])), Point(round(lineTemp[1][0]), round(lineTemp[1][1]))))
    checkedLines.append(lineIdx)


    for lineC in range(len(lines)):
        checkLine = lines[lineC]
        samePoint = -1
        if isSamePoint(line[otherSideIdx], checkLine.p1, OUTPUT_POINT_SIMILARITY_COMPARE_AREA_RADIUS):
            samePoint = 0
        elif isSamePoint(line[otherSideIdx], checkLine.p2, OUTPUT_POINT_SIMILARITY_COMPARE_AREA_RADIUS):
            samePoint = 1
        
        if samePoint != -1:
            followLine(lines, lineC, 1-samePoint, component_endpoints, checkedLines, outputLines, horizontalCount)


def createDirectoryTree():
    filenames = ["output/_rels/", "output/circuitdiagram/", "output/docProps/"]
    for filename in filenames:
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except:
                print("Cannot create directories for output!")

def findIntersection(line1, line2):
    x1 = line1.p1.x
    x2 = line1.p2.x
    y1 = line1.p1.y
    y2 = line1.p2.y
    x3 = line2.p1.x
    x4 = line2.p2.x
    y3 = line2.p1.y
    y4 = line2.p2.y

    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )

    return Point(round(px), round(py))





# ----- MAIN -----


# load in CNN model
t1 = time.perf_counter()

if len(sys.argv) < 2:
    exit("Not enough parameter given!")

if not os.path.isfile(str(sys.argv[1])):
    exit("The file does not exist!")

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(gray, 7)

thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]

thresh = 255 - thresh

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

if len(sys.argv) > 2:
    if 5 < int(sys.argv[2]) or int(sys.argv[2]) < 0:
        exit("Canvas number is out of range! The number must be between 0 and 5!")
    
    img = putOnCanvas(img, 100 - int(sys.argv[2]) * 10)
    thresh = putOnCanvas(thresh, 100 - int(sys.argv[2]) * 10)

biggerSide = img.shape[0] if img.shape[0] > img.shape[1] else img.shape[1]

POINT_SIMILARITY_COMPARE_AREA_RADIUS = round(biggerSide*0.00375)
LINE_MIN_LENGTH = round(biggerSide*0.035)
LINE_CHECK_SIMILARITY_THRESHOLD = round(biggerSide*0.00375)
COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH = round(biggerSide*0.0125)
COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH = round(biggerSide*0.0875)
COMPONENT_MIN_BOX_SIZE = round(biggerSide*0.05)
COMPONENT_BOX_SIZE_OFFSET = round(biggerSide*0.015)
OUTPUT_POINT_SIMILARITY_COMPARE_AREA_RADIUS = round(biggerSide*0.0075)


canny = cv2.Canny(thresh, 100, 150)
canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

linesP = list(cv2.HoughLinesP(canny, 1, np.pi/180, LINE_MIN_LENGTH, None, LINE_MIN_LENGTH, 0))
if linesP is None:
    exit("No lines detected")

linesP = convertHoughToLineList(linesP)

topLine = Line(Point(0, img.shape[0]), Point(0, img.shape[0]))
botLine = Line(Point(0, 0), Point(0, 0))
leftLine = Line(Point(img.shape[1], 0), Point(img.shape[1], 0))
rightLine = Line(Point(0, 0), Point(0, 0))

topValue = img.shape[0]
botValue = 0
leftValue = img.shape[1]
rightValue = 0


for l in linesP:
    l_left, l_right = ((l.p1, l.p2) if l.p1.x < l.p2.x else (l.p2, l.p1))
    l_top, l_bot = ((l.p1, l.p2) if l.p1.y < l.p2.y else (l.p2, l.p1))
    l_middle = Point(round(l_left.x+((l_right.x-l_left.x)/2)), round(l_top.y+((l_bot.y-l_top.y)/2)))
    
    if abs(l.p2.x-l.p1.x) > abs(l.p2.y-l.p1.y):
        # horizontal
        if l_middle.y < topValue:
            topValue = l_middle.y
            topLine = Line(Point(l_left.x, l_left.y), Point(l_right.x, l_right.y))
        elif l_middle.y > botValue:
            botValue = l_middle.y
            botLine = Line(Point(l_left.x, l_left.y), Point(l_right.x, l_right.y))
    else:
        # vertical
        if l_middle.x < leftValue:
            leftValue = l_middle.x
            leftLine = Line(Point(l_left.x, l_left.y), Point(l_right.x, l_right.y))
        elif l_middle.x > rightValue:
            rightValue = l_middle.x
            rightLine = Line(Point(l_left.x, l_left.y), Point(l_right.x, l_right.y))

cv2.line(img, (topLine.p1.x, topLine.p1.y), (topLine.p2.x, topLine.p2.y), (255,0,255), thickness=5)
cv2.line(img, (botLine.p1.x, botLine.p1.y), (botLine.p2.x, botLine.p2.y), (255,0,255), thickness=5)
cv2.line(img, (leftLine.p1.x, leftLine.p1.y), (leftLine.p2.x, leftLine.p2.y), (255,0,255), thickness=5)
cv2.line(img, (rightLine.p1.x, rightLine.p1.y), (rightLine.p2.x, rightLine.p2.y), (255,0,255), thickness=5)


# cv2.circle(img, (intersection.x, intersection.y), 15, (255,0,255), thickness=-1)
tl = findIntersection(topLine, leftLine)
bl = findIntersection(botLine, leftLine)
tr = findIntersection(topLine, rightLine)
br = findIntersection(botLine, rightLine)

pts1 = np.float32([[tl.x, tl.y],[tr.x, tr.y],[br.x, br.y],[bl.x, bl.y]])

w = getLineLength(Line(Point(pts1[0][0], pts1[0][1]), Point(pts1[1][0], pts1[1][1])))
h = getLineLength(Line(Point(pts1[0][0], pts1[0][1]), Point(pts1[3][0], pts1[3][1])))

offset = 500
pts2 = np.float32([[offset, offset],[offset+w, offset],[offset+w, offset+h],[offset, offset+h]])

M = cv2.getPerspectiveTransform(pts1,pts2)
img = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]))
thresh = cv2.warpPerspective(thresh,M,(img.shape[1], img.shape[0]))


cv2.imshow("img", resizeImage(img, PICTURE_SCALE))
cv2.imshow("thresh", resizeImage(thresh, PICTURE_SCALE))

cv2.waitKey(0)