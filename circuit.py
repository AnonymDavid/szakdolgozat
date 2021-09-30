import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, NamedTuple
from enum import Enum
import math
import sys
import os.path
import random
from datetime import datetime

# from tensorflow import keras

# temporary imports
import time

# ----- CONSTANTS ----- TODO: all should be percent?
CNT_DELETE_PERCENT = 15
POINT_SIMILARITY_COMPARE_AREA_RADIUS = 10
LINE_COUNT_CHECK_FOR_ROTATION = 50
LINE_SEARCH_ANGLE_THRESHOLD = 5 # degrees, both ways
LINE_CHECK_SIMILARITY_THRESHOLD = 15
LINE_AGGREGATION_SIMILARITY_THRESHOLD = 25

# temp:
PICTURE_SCALE = 25




# ----- TYPES -----
class Orientation(Enum):
    HORIZONTAL_LEFT = 0
    HORIZONTAL_RIGHT = 1
    VERTICAL_TOP = 2
    VERTICAL_BOT = 3

class Point(NamedTuple):
    x: int
    y: int

@dataclass
class Line:
    p1: Point
    p2: Point

@dataclass
class Endpoint:
    point: Point
    orientation: Orientation

@dataclass
class Intersection:
    point: Point
    lineCount: int



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

def resizeImage(src, percent: int):
    return cv2.resize(src, (int(src.shape[1]*percent/100), int(src.shape[0]*percent/100)))
    




# ----- MAIN -----

t1 = time.perf_counter()

if not len(sys.argv) == 2:
    exit("No parameter given!")

if not os.path.isfile(str(sys.argv[1])):
    exit("The file does not exist!")

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(gray, 7)

thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]

thresh = 255 - thresh

# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=3)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=3)


# ROTATE VIA LONGEST LINES AVG ANGLE
linesP = list(cv2.HoughLinesP(thresh, 1, np.pi/180, 100, None, 50, 0))

if linesP is None:
    exit("No lines detected")

linesP.sort(key=lambda l: getLineLength(convertHoughToLine(l)), reverse=True )

avgAngleDiff = 0
linesP = convertHoughToLineList(linesP)

rotatelines = []
checkedLineCount = 1
differentLineCount = 1
rotatelines.append(linesP[0])

while checkedLineCount < len(linesP) and differentLineCount < LINE_COUNT_CHECK_FOR_ROTATION:
    line = linesP[checkedLineCount]
    i = 1
    while i < differentLineCount and not isSameLine(line, rotatelines[i]):
        i += 1
    if (i >= differentLineCount):
        angle = getLineAngle(line)
        angle = angle - (int)(angle / 90)*90
        if (angle > 45):
            angle = angle - 90
        
        currentDiffWithAvg = abs((avgAngleDiff/differentLineCount) - angle)
        
        if abs(currentDiffWithAvg - int(currentDiffWithAvg / 90) * 90) <= 20:
            avgAngleDiff += angle
            rotatelines.append(line)
            differentLineCount += 1

    checkedLineCount += 1
    
avgAngleDiff /= differentLineCount

thresh = rotateImage(thresh, avgAngleDiff)
gray = rotateImage(gray, avgAngleDiff)
img = rotateImage(img, avgAngleDiff)

# TODO: houghlinesp parameters with percent
linesP = list(cv2.HoughLinesP(thresh, 1, np.pi/180, 100, None, 120, 0))
if linesP is None:
    exit("No lines detected")

linesP = convertHoughToLineList(linesP)

horizontal = []
vertical = []
for line in linesP:
    line_angle = getLineAngle(line)
    if abs(line_angle - (round(line_angle / 180) * 180)) < LINE_SEARCH_ANGLE_THRESHOLD:
        line_left, line_right = ((line.p1, line.p2) if line.p1.x < line.p2.x else (line.p2, line.p1))
        lineC = 0
        while lineC < len(horizontal):
            hl_left, hl_right = ((horizontal[lineC].p1, horizontal[lineC].p2) if horizontal[lineC].p1.x < horizontal[lineC].p2.x else (horizontal[lineC].p2, horizontal[lineC].p1))
            line_middle_y = line_left.y if line_left.y < line_right.y else line_right.y + abs(line_left.y - line_right.y) / 2
            if line_left.x <= hl_right.x and line_right.x >= hl_left.x:
                hl_middle_y = hl_left.y if hl_left.y < hl_right.y else hl_right.y + abs(hl_left.y - hl_right.y) / 2

                if abs(hl_middle_y - line_middle_y) < LINE_AGGREGATION_SIMILARITY_THRESHOLD:
                    tempLine = Line((line_left if line_left.x < hl_left.x else hl_left), (line_right if line_right.x > hl_right.x else hl_right))
                    tempLineAngle = getLineAngle(tempLine)
                    if tempLineAngle < LINE_SEARCH_ANGLE_THRESHOLD or 360-tempLineAngle < LINE_SEARCH_ANGLE_THRESHOLD:
                        horizontal[lineC] = tempLine
                        break

            lineC += 1
        
        if lineC >= len(horizontal):
            horizontal.append(line)
    elif abs(line_angle - (round(line_angle / 90) * 90)) < LINE_SEARCH_ANGLE_THRESHOLD:
        line_top, line_bottom = ((line.p1, line.p2) if line.p1.y < line.p2.y else (line.p2, line.p1))
        line_middle_x = line_top.x if line_top.x < line_bottom.x else line_bottom.x + abs(line_top.x - line_bottom.x) / 2
        lineC = 0
        while lineC < len(vertical):
            hl_top, hl_bottom = ((vertical[lineC].p1, vertical[lineC].p2) if vertical[lineC].p1.y < vertical[lineC].p2.y else (vertical[lineC].p2, vertical[lineC].p1))
            if line_top.y <= hl_bottom.y and line_bottom.y >= hl_top.y:
                hl_middle_x = hl_top.x if hl_top.x < hl_bottom.x else hl_bottom.x + abs(hl_top.x - hl_bottom.x) / 2

                if abs(hl_middle_x - line_middle_x) < LINE_AGGREGATION_SIMILARITY_THRESHOLD:
                    tempLine = Line((line_top if line_top.y < hl_top.y else hl_top), (line_bottom if line_bottom.y > hl_bottom.y else hl_bottom))
                    tempLineAngle = getLineAngle(tempLine)
                    if abs(tempLineAngle - 90) < LINE_SEARCH_ANGLE_THRESHOLD:
                        vertical[lineC] = tempLine
                        break

            lineC += 1
        
        if lineC >= len(vertical):
            vertical.append(line)

lines = horizontal + vertical

endpoints = []

ep_HL = []
ep_HR = []
ep_VT = []
ep_VB = []


for line in horizontal:
    if line.p1.x < line.p2.x:
        endpoints.append(Endpoint(line.p1, Orientation.HORIZONTAL_LEFT))
        endpoints.append(Endpoint(line.p2, Orientation.HORIZONTAL_RIGHT))

        ep_HL.append(line.p1)
        ep_HR.append(line.p2)
    else:
        endpoints.append(Endpoint(line.p1, Orientation.HORIZONTAL_RIGHT))
        endpoints.append(Endpoint(line.p2, Orientation.HORIZONTAL_LEFT))
        
        ep_HL.append(line.p2)
        ep_HR.append(line.p1)

for line in vertical:
    if line.p1.y < line.p2.y:
        endpoints.append(Endpoint(line.p1, Orientation.VERTICAL_TOP))
        endpoints.append(Endpoint(line.p2, Orientation.VERTICAL_BOT))

        ep_VT.append(line.p1)
        ep_VB.append(line.p2)
    else:
        endpoints.append(Endpoint(line.p1, Orientation.VERTICAL_BOT))
        endpoints.append(Endpoint(line.p2, Orientation.VERTICAL_TOP))

        ep_VT.append(line.p2)
        ep_VB.append(line.p1)


for l in lines:
    cv2.line(img, (l.p1.x, l.p1.y), (l.p2.x, l.p2.y), (0,255,0), 4)

for ep in endpoints:
    cv2.circle(img, (ep.point.x, ep.point.y), 12, (255,0,255), -1)

cv2.line(img, (10, 10), (40, 10), (0,255,0), 4)






# # CONNECTED COMPONENTS
# findAreaTempImg = 255 - thresh

# # finding areas
# num_labels, labels_im = cv2.connectedComponents(findAreaTempImg)

# areas = []

# for label in range(1,num_labels):
#     # create mask from area
#     mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
#     mask[labels_im == label] = 255
    
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     cnt = contours[0]
    
#     # figuring out shapes
#     peri = cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
#     x, y, w, h = cv2.boundingRect(approx)
    
#     areas.append([x,y,w,h])
    
#     # if two areas overlap, delete the bigger (filter out ares where components and lines enclosed an area)
#     for i in range(len(areas) - 1):
#         if (x <= areas[i][0]+areas[i][2]) and (y <= areas[i][1]+areas[i][3]) and (x+w >= areas[i][0]) and (y+h >= areas[i][1]):
#             if (w*h < areas[i][2]*areas[i][3]):
#                 del(areas[i])
#             else:
#                 del(areas[-1])
#             break

# for area in areas:
#     cv2.rectangle(img, (area[0], area[1]), (area[0]+area[2], area[1]+area[3]), (0,0,255), 8)





# ============================== CNN ====================================
# model = keras.models.load_model("symbolsModel.h5")

# # thresh = cv2.resize(thresh, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
# prediction = model.predict(thresh)
# if np.argmax(prediction[0]) == 0:
#     print("inductor")
# else:
#     print("resistor")


cv2.imshow("img", resizeImage(img, PICTURE_SCALE))
cv2.imshow("thresh", resizeImage(thresh, PICTURE_SCALE))

t2 = time.perf_counter()

print(f"\nTime: {t2 - t1:0.4f} seconds")

cv2.waitKey(0)