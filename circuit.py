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

from tensorflow import keras

# temporary imports
import time

# ----- CONSTANTS ----- #TODO: all should be percent?
CNT_DELETE_PERCENT = 15
POINT_SIMILARITY_COMPARE_AREA_RADIUS = 10
LINE_COUNT_CHECK_FOR_ROTATION = 10
LINE_SEARCH_ANGLE_THRESHOLD = 5 # degrees, both ways
LINE_CHECK_SIMILARITY_THRESHOLD = 15
LINE_AGGREGATION_SIMILARITY_THRESHOLD = 25
COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH = 50
COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH = 300
COMPONENT_OTHER_ENDPOINT_SEARCH_MIN_LENGTH = 16
COMPONENT_MIN_BOX_SIZE = 150
COMPONENT_BOX_SIZE_OFFSET = 40
OUTPUT_SCALE = 50

# temp:
PICTURE_SCALE = 25




# ----- TYPES -----
class Orientation(Enum):
    HORIZONTAL_LEFT = 0
    HORIZONTAL_RIGHT = 1
    VERTICAL_TOP = 2
    VERTICAL_BOT = 3
    HORIZONTAL = 4
    VERTICAL = 5

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

def pointsCloseArray(p1: Point, points: List[Point]) -> int:
    """Check if the given point is close to another in the array."""
    for i in range(len(points)):
        if isSamePoint(p1, points[i]):
            return i
    return -1

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


# load in CNN model
model = keras.models.load_model("symbolsModel.h5")


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


# finding connection lines
linesP = list(cv2.HoughLinesP(thresh, 1, np.pi/180, 100, None, 110, 0))
if linesP is None:
    exit("No lines detected")

linesP = convertHoughToLineList(linesP)

# filtering and separating horizontal and vertical lines
horizontal = []
vertical = []
for line in linesP:
    line_angle = getLineAngle(line)
    # horizontal
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
    # vertical
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

# finding and separating line endpoints

ep_HL = []
ep_HR = []
ep_VT = []
ep_VB = []

for line in horizontal:
    if line.p1.x < line.p2.x:
        ep_HL.append(line.p1)
        ep_HR.append(line.p2)
    else:
        ep_HL.append(line.p2)
        ep_HR.append(line.p1)

for line in vertical:
    if line.p1.y < line.p2.y:
        ep_VT.append(line.p1)
        ep_VB.append(line.p2)
    else:
        ep_VT.append(line.p2)
        ep_VB.append(line.p1)


# for l in lines:
#     cv2.line(img, (l.p1.x, l.p1.y), (l.p2.x, l.p2.y), (0,255,0), 4)

# for ep in ep_HL:
#     cv2.circle(img, (ep.x, ep.y), 12, (255,0,255), -1)
# for ep in ep_HR:
#     cv2.circle(img, (ep.x, ep.y), 12, (255,0,255), -1)
# for ep in ep_VT:
#     cv2.circle(img, (ep.x, ep.y), 12, (255,0,255), -1)
# for ep in ep_VB:
#     cv2.circle(img, (ep.x, ep.y), 12, (255,0,255), -1)


# finding components

solo_ep_HL = []
solo_ep_HR = []
solo_ep_VT = []
solo_ep_VB = []

components = []
compCount = 0

# horizontal components
for hr in ep_HR:
    # cv2.rectangle(img, (hr.x + COMPONENT_OTHER_ENDPOINT_SEARCH_MIN_LENGTH, hr.y - round(COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2)), (hr.x + COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH, hr.y + round(COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2)), (0,0,255), 5)
    hlc = compCount
    while (hlc < len(ep_HL) and
        (
            ep_HL[hlc].y > hr.y + COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            ep_HL[hlc].y < hr.y - COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            ep_HL[hlc].x > hr.x + COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH or
            ep_HL[hlc].x < hr.x + COMPONENT_OTHER_ENDPOINT_SEARCH_MIN_LENGTH
        )):
        hlc += 1
    
    if hlc < len(ep_HL):
        compSize = ep_HL[hlc].x - hr.x
        if compSize < COMPONENT_MIN_BOX_SIZE:
            compSize = COMPONENT_MIN_BOX_SIZE
        components.append([Point(hr.x - COMPONENT_BOX_SIZE_OFFSET, hr.y - round(compSize/2)), Point(ep_HL[hlc].x + COMPONENT_BOX_SIZE_OFFSET, ep_HL[hlc].y + round(compSize / 2)), Orientation.HORIZONTAL])
        ep_HL[compCount], ep_HL[hlc] = ep_HL[hlc], ep_HL[compCount]
        compCount += 1
    else:
        solo_ep_HR.append(hr)

for i in range(compCount, len(ep_HL)):
    solo_ep_HL.append(ep_HL[i])
    

# vertical components
compCount = 0
for vb in ep_VB:
    # cv2.rectangle(img, (vb.x - round(COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2), vb.y + COMPONENT_OTHER_ENDPOINT_SEARCH_MIN_LENGTH), (vb.x + round(COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2), vb.y + COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH), (255,0,0), 8)
    vtc = compCount
    
    while (vtc < len(ep_VT) and
        (
            ep_VT[vtc].x > vb.x + COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            ep_VT[vtc].x < vb.x - COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            ep_VT[vtc].y > vb.y + COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH or
            ep_VT[vtc].y < vb.y + COMPONENT_OTHER_ENDPOINT_SEARCH_MIN_LENGTH
        )):
        vtc += 1
    
    if vtc < len(ep_VT):
        compSize = ep_VT[vtc].y - vb.y
        if compSize < COMPONENT_MIN_BOX_SIZE:
            compSize = COMPONENT_MIN_BOX_SIZE
        
        components.append([Point(vb.x - round(compSize/2), vb.y - COMPONENT_BOX_SIZE_OFFSET), Point(ep_VT[vtc].x + round(compSize / 2), ep_VT[vtc].y + COMPONENT_BOX_SIZE_OFFSET), Orientation.VERTICAL])
        ep_VT[compCount], ep_VT[vtc] = ep_VT[vtc], ep_VT[compCount]
        compCount += 1
    else:
        solo_ep_VB.append(vb)

for i in range(compCount, len(ep_VT)):
    solo_ep_VT.append(ep_VT[i])


# start output file
file = open("Document.xml", "w", newline='')
file.write(
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<circuit version="1.4" xmlns="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document">\n'
	'\t<properties>\n'
    '\t\t<width>'+str(img.shape[1])+'</width>\n'
    '\t\t<height>'+str(img.shape[0])+'</height>\n'
    '\t</properties>\n'
)
file.write(
    '\t<definitions>\n'
	'\t\t<src col="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/components/common">\n'
	'\t\t\t<add id="0" item="lamp" p4:guid="c0da4646-ffaf-48f0-8a56-2e8e07148ecf" p4:version="3.0.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t\t<add id="1" item="acsource" p4:guid="8979c617-b041-4069-8769-0b76dc0e6c52" p4:version="1.2.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t\t<add id="2" item="cell" p4:guid="8979c617-b041-4069-8769-0b76dc0e6c52" p4:version="1.2.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t\t<add id="3" item="transnpn" p4:guid="247a790e-c7aa-43ca-9e0c-c524257c0cd5" p4:version="1.0.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t\t<add id="4" item="transpnp" p4:guid="247a790e-c7aa-43ca-9e0c-c524257c0cd5" p4:version="1.0.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t\t<add id="5" item="inductor" p4:guid="e6ebc838-fb6a-4078-8d03-7fed04cc6938" p4:version="2.0.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t\t<add id="6" item="battery" p4:guid="8979c617-b041-4069-8769-0b76dc0e6c52" p4:version="1.2.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t\t<add id="7" item="toggleswitch" p4:guid="421694bf-755a-4d50-b2ca-7273216ca17d" p4:version="1.0.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t\t<add id="8" item="capacitor" p4:guid="154b7831-58e6-4126-b582-8c1cc2bdeb13" p4:version="4.0.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t\t<add id="9" item="resistor" p4:guid="dab6d52b-51a0-49b0-bc40-3cda966148aa" p4:version="4.0.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t</src>\n'
	'\t\t<src>\n'
	'\t\t\t<add id="10" item="Standard Diode" p4:guid="099c5645-f7c6-4933-bed9-b0f4aa3f4a2a" p4:version="1.0.0" xmlns:p4="http://schemas.circuit-diagram.org/circuitDiagramDocument/2012/document/component-descriptions" />\n'
	'\t\t</src>\n'
	'\t</definitions>\n'
	'\t<elements>\n'
)

componentId = 0
# find third/forth connection on components if exists (transistor)
for c in components:
    compMiddleX = round(c[0].x + ((c[1].x - c[0].x) / 2))
    compMiddleY = round(c[0].y + ((c[1].y - c[0].y) / 2))
    if c[2] == Orientation.HORIZONTAL:
        # top side
        vbc = 0
        while (vbc < len(solo_ep_VB) and
        (
            solo_ep_VB[vbc].x > compMiddleX + COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            solo_ep_VB[vbc].x < compMiddleX - COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            solo_ep_VB[vbc].y < compMiddleY - COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH or
            solo_ep_VB[vbc].y > compMiddleY - COMPONENT_OTHER_ENDPOINT_SEARCH_MIN_LENGTH
        )):
            vbc += 1
    
        if vbc < len(solo_ep_VB):
            c[0] = Point(c[0].x, solo_ep_VB[vbc].y - COMPONENT_BOX_SIZE_OFFSET*2)
        
        # bot side
        vtc = 0
        while (vtc < len(solo_ep_VT) and
        (
            solo_ep_VT[vtc].x > compMiddleX + COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            solo_ep_VT[vtc].x < compMiddleX - COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            solo_ep_VT[vtc].y > compMiddleY + COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH or
            solo_ep_VT[vtc].y < compMiddleY + COMPONENT_OTHER_ENDPOINT_SEARCH_MIN_LENGTH
        )):
            vtc += 1
    
        if vtc < len(solo_ep_VT):
            c[1] = Point(c[1].x, solo_ep_VT[vtc].y + COMPONENT_BOX_SIZE_OFFSET*2)
    else:
        # right side
        hlc = 0
        while (hlc < len(solo_ep_HL) and
        (
            solo_ep_HL[hlc].y > compMiddleY + COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            solo_ep_HL[hlc].y < compMiddleY - COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            solo_ep_HL[hlc].x > compMiddleX + COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH or
            solo_ep_HL[hlc].x < compMiddleX + COMPONENT_OTHER_ENDPOINT_SEARCH_MIN_LENGTH
        )):
            hlc += 1
    
        if hlc < len(solo_ep_HL):
            c[1] = Point(solo_ep_HL[hlc].x + COMPONENT_BOX_SIZE_OFFSET*2, c[1].y)
        
        # left side
        hrc = 0
        while (hrc < len(solo_ep_HR) and
        (
            solo_ep_HR[hrc].y > compMiddleY + COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            solo_ep_HR[hrc].y < compMiddleY - COMPONENT_OTHER_ENDPOINT_SEARCH_WIDTH/2 or
            solo_ep_HR[hrc].x < compMiddleX - COMPONENT_OTHER_ENDPOINT_SEARCH_MAX_LENGTH or
            solo_ep_HR[hrc].x > compMiddleX - COMPONENT_OTHER_ENDPOINT_SEARCH_MIN_LENGTH
        )):
            hrc += 1
        
        if hrc < len(solo_ep_HR):
            c[0] = Point(solo_ep_HR[hrc].x - COMPONENT_BOX_SIZE_OFFSET*2, c[0].y)
    
    
    cv2.rectangle(img, (c[0][0], c[0][1]), (c[1][0], c[1][1]), (0,0,255), 5)

    componentImg = thresh[c[0][1]:c[1][1], c[0][0]:c[1][0]]

    if c[2] == Orientation.VERTICAL:
        componentImg = rotateImage(componentImg, 90)

    cv2.waitKey(0)

    componentImg = cv2.resize(componentImg, dsize=(150,150), interpolation=cv2.INTER_CUBIC)
    componentImg = componentImg.reshape(-1, 150, 150, 1)
    
    # predict component with CNN
    prediction = model.predict(componentImg)
    bestPrediction = np.argmax(prediction[0])
    bestPredictionValue = prediction[0][bestPrediction]
    
    componentImg = np.flip(componentImg, 1)
    
    prediction = model.predict(componentImg)

    if prediction[0][np.argmax(prediction[0])] > bestPredictionValue:
        bestPrediction = np.argmax(prediction[0])

    if bestPrediction == 0:
        cv2.putText(img, "battery", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{6}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
            '\t\t\t\t<p k="voltage" v="5" />\n'
            '\t\t\t\t<p k="frequency" v="50" />\n'
            '\t\t\t\t<p k="showvoltage" v="False" />\n'
            '\t\t\t\t<p k="t" v="Battery" />\n'
            '\t\t\t</prs>\n'
            '\t\t\t<cns />\n'
		    '\t\t</c>\n'
        )
    elif bestPrediction == 1:
        cv2.putText(img, "capacitor", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{8}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
            '\t\t\t\t<p k="style" v="IEC" />\n'
            '\t\t\t\t<p k="text" v="" />\n'
            '\t\t\t\t<p k="capacitance" v="0.00000" />\n'
            '\t\t\t\t<p k="showCapacitance" v="False" />\n'
            '\t\t\t\t<p k="t" v="Standard" />\n'
            '\t\t\t</prs>\n'
            '\t\t<cns />\n'
		    '\t\t</c>\n'
        )
    elif bestPrediction == 2:
        cv2.putText(img, "diode", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{10}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
            '\t\t\t</prs>\n'
            '\t\t\t<cns />\n'
		    '\t\t</c>\n'
        )
    elif bestPrediction == 3:
        cv2.putText(img, "inductor", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{5}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
            '\t\t\t\t<p k="text" v="" />\n'
			'\t\t\t\t<p k="inductance" v="0.1" />\n'
			'\t\t\t\t<p k="showInductance" v="False" />\n'
            '\t\t\t</prs>\n'
            '\t\t\t<cns />\n'
		    '\t\t</c>\n'
        )
    elif bestPrediction == 4:
        cv2.putText(img, "lamp", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{0}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
            '\t\t\t\t<p k="t" v="Generic" />\n'
			'\t\t\t\t<p k="islit" v="False" />\n'
            '\t\t\t</prs>\n'
            '\t\t\t<cns />\n'
		    '\t\t</c>\n'
        )
    elif bestPrediction == 5:
        cv2.putText(img, "resistor", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{9}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
			'\t\t\t\t<p k="style" v="IEC" />\n'
			'\t\t\t\t<p k="text" v="" />\n'
			'\t\t\t\t<p k="resistance" v="1" />\n'
			'\t\t\t\t<p k="showResistance" v="False" />\n'
			'\t\t\t\t<p k="t" v="Standard" />\n'
            '\t\t\t</prs>\n'
            '\t\t\t<cns />\n'
		    '\t\t</c>\n'
        )
    elif bestPrediction == 6:
        cv2.putText(img, "source_ac", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{1}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
			'\t\t\t\t<p k="voltage" v="1" />\n'
			'\t\t\t\t<p k="frequency" v="1" />\n'
			'\t\t\t\t<p k="showvoltage" v="True" />\n'
			'\t\t\t\t<p k="t" v="AC" />\n'
            '\t\t\t</prs>\n'
            '\t\t\t<cns />\n'
		    '\t\t</c>\n'
        )
    elif bestPrediction == 7:
        cv2.putText(img, "source_dc", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{2}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
			'\t\t\t\t<p k="voltage" v="1" />\n'
			'\t\t\t\t<p k="frequency" v="1" />\n'
			'\t\t\t\t<p k="showvoltage" v="True" />\n'
			'\t\t\t\t<p k="t" v="DC" />\n'
            '\t\t\t</prs>\n'
            '\t\t\t<cns />\n'
		    '\t\t</c>\n'
        )
    elif bestPrediction == 8:
        cv2.putText(img, "switch", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{7}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
			'\t\t\t\t<p k="t" v="Toggle" />\n'
			'\t\t\t\t<p k="closed" v="False" />\n'
            '\t\t\t</prs>\n'
            '\t\t\t<cns />\n'
		    '\t\t</c>\n'
        )
    elif bestPrediction == 9:
        cv2.putText(img, "transistor_npn", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{3}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
			'\t\t\t\t<p k="t" v="NPN" />\n'
            '\t\t\t</prs>\n'
            '\t\t\t<cns />\n'
		    '\t\t</c>\n'
        )
    elif bestPrediction == 10:
        cv2.putText(img, "transistor_pnp", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=3)
        file.write(
            '\t\t<c id="'+str(componentId)+'" tp="{4}" x="'+str(c[0][0])+'" y="'+str(c[0][1])+'" o="'+('h' if c[2] == Orientation.HORIZONTAL else 'v')+'" sz="'+str(c[1][0]-c[0][0] if c[2] == Orientation.HORIZONTAL else c[1][1]-c[0][1])+'" flp="false">\n'
            '\t\t\t<prs>\n'
			'\t\t\t\t<p k="t" v="PNP" />\n'
            '\t\t\t</prs>\n'
            '\t\t\t<cns />\n'
		    '\t\t</c>\n'
        )

    componentId += 1

    # cv2.imshow("component", cv2.resize(componentImg, (150,150)))
    # cv2.waitKey(0)



# finish output file
file.write(
    '\t</elements>\n'
    '</circuit>\n'
)

file.close()









cv2.imshow("img", resizeImage(img, PICTURE_SCALE))
cv2.imshow("thresh", resizeImage(thresh, PICTURE_SCALE))

t2 = time.perf_counter()

print(f"\nTime: {t2 - t1:0.4f} seconds")

cv2.waitKey(0)