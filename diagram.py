import numpy as np
import cv2

import sys
import os.path


def pointsClose(pt1, pt2, threshold):
    if abs(pt1[0] - pt2[0]) < threshold and abs(pt1[1] - pt2[1]) < threshold:
        return True
    return False

def pointInRect(pt, rect):
    rpt1 = (rect[0], rect[1])
    rpt2 = (rect[0]+rect[2], rect[1]+rect[3])
    
    if (pt[0]<=rpt2[0]) and (pt[1]<=rpt2[1]) and (pt[0]>=rpt1[0]) and (pt[1]>=rpt1[1]):
        return True
    return False

def pointCloseToRect(pt, rect, threshold):
    mrect = (rect[0]-threshold, rect[1]-threshold, rect[2]+threshold*2, rect[3]+threshold*2)
    return pointInRect(pt, mrect)

def isIntersection(pt, intersections, threshold):
    i = 0
    while i < len(intersections) and not pointsClose(pt, intersections[i][0], threshold):
        i += 1
    
    if (i < len(intersections)):
        return i
    return -1

def getIntersectionLines(pt, lines, threshold):
    iLines = []
    for i in range(len(lines)):
        if pointsClose(pt, lines[i][0], threshold):
            iLines.append([lines[i], 0, i])
        elif pointsClose(pt, lines[i][1], threshold):
            iLines.append([lines[i], 1, i])
    
    return iLines

def getIntersectionEndpoints(ISidx, intersections, lines, endpoints, threshold, inspectedIS = []):
    iln = getIntersectionLines(intersections[ISidx][0], lines, threshold)
    ep = []
    for i in range(len(iln)):
        if iln[i][1] == 0:
            otherSide = iln[i][0][1]
        else:
            otherSide = iln[i][0][0]
        
        
        otherSideLineIdx = iln[i][2]
        
        j = 0
        while j < len(lines):
            i = isIntersection(otherSide, intersections, threshold)
            if i >= 0:
                x = 0
                while x < len(inspectedIS) and not inspectedIS[x] == i:
                    x += 1
                if x < len(inspectedIS):
                    epTemp, inspectedIS = getIntersectionEndpoints(isIntersection(otherSide, intersections, threshold), intersections, lines, endpoints, threshold, inspectedIS)
                    for e in epTemp:
                        ep.append(e)
                    otherSide = (-1, -1)
            elif not j == otherSideLineIdx:
                if pointsClose(lines[j][0], otherSide, threshold):
                    otherSide = lines[j][1]
                    otherSideLineIdx = j
                    j = -1
                elif pointsClose(lines[j][1], otherSide, threshold):
                    otherSide = lines[j][0]
                    otherSideLineIdx = j
                    j = -1
            j += 1
        
        if otherSide[0] >= 0:
            y = 0
            while y < len(endpoints) and not endpoints[y][0] == otherSide:
                y += 1
            if y < len(endpoints):
                ep.append(otherSide)
    
    inspectedIS.append(ISidx)
    
    return ep, inspectedIS;

def getPointOrientation(pt, rect):
    center = (rect[0]+round(rect[2]/2), rect[1]+round(rect[3]/2))
    
    if (pt[0] <= center[0]-15):
        return 'L'
    elif (pt[0] >= center[0]+15):
        return 'R'
    elif (pt[1] <= center[1]-15):
        return 'T'
    else:
        return 'B'


def getOrientationValues(orientation):
    if orientation == 'L':
        return (0, 0.5)
    elif orientation == 'R':
        return (1, 0.5)
    elif orientation == 'T':
        return (0.5, 0)
    elif orientation == 'B':
        return (0.5, 1)




# read input image, preprocessing
if not len(sys.argv) == 2:
    print("The program needs exactly 1 argument!")
    exit()

if not os.path.isfile(str(sys.argv[1])):
    print("The file does not exist!")
    exit()

img = cv2.imread(str(sys.argv[1]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Original image", img)

kernel = np.ones((5,5), np.uint8)
crosskernel = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]], dtype = np.uint8)

thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
eroded = cv2.erode(thresh, kernel)

# finding areas
num_labels, labels_im = cv2.connectedComponents(eroded)

areas = []

for label in range(1,num_labels):
    # create mask from area
    mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    mask[labels_im == label] = 255
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnt = contours[0]
    
    # figuring out shapes
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
    x, y, w, h = cv2.boundingRect(approx)
    
    if len(approx) == 3:
        # triangle
        areas.append([[x,y,w,h],'T',cnt])
    elif len(approx) == 4:
        # quadrangle
        if 1 - (cv2.contourArea(cnt) / (w*h)) <= 0.05:
            # rectangle
            areas.append([[x,y,w,h],'R',cnt])
        elif abs(1 - (2 * cv2.contourArea(cnt) / (w*h))) <= 0.05:
            # diamond
            areas.append([[x,y,w,h],'D',cnt])
        else:
            # parallelogram
            areas.append([[x,y,w,h],'P',cnt])
    elif len(approx) > 4:
        # elipse
        areas.append([[x,y,w,h],'E',cnt])
    
    # if two areas overlap, delete the bigger (filter out ares where components and lines enclosed an area)
    for i in range(len(areas) - 1):
        if (x <= areas[i][0][0]+areas[i][0][2]) and (y <= areas[i][0][1]+areas[i][0][3]) and (x+w >= areas[i][0][0]) and (y+h >= areas[i][0][1]):
            if (w*h < areas[i][0][2]*areas[i][0][3]):
                del(areas[i])
            else:
                del(areas[-1])
            break


isArrowdetectionSuccessful = False

try:
    # making full a mask for every shape and arrow
    full = thresh.copy()
    
    for area in areas:
        cv2.drawContours(full, [area[2]], -1, 0, -1)

    full = 255 - full
    
    full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, kernel)

    # making fullcontours a mask for only the shapes (covering the outer borders)
    fullcontours = full

    # erode until there's as many contours as shapes
    contours, hierarchy = cv2.findContours(fullcontours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    erodeCount = 0
    while not (len(contours) == len(areas)):
        fullcontours = cv2.erode(fullcontours,crosskernel)
        fullcontours = cv2.morphologyEx(fullcontours, cv2.MORPH_OPEN, kernel)
        erodeCount += 1
        contours, hierarchy = cv2.findContours(fullcontours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fullcontours = cv2.morphologyEx(fullcontours, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(fullcontours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # draw every contour to cover the shapes
    for cnt in contours:
        cv2.drawContours(fullcontours, [cnt], -1, 255, -1)
        cv2.drawContours(fullcontours, [cnt], -1, 255, (erodeCount*2))
    
    # making arrows a mask containing only the arrows
    arrows = cv2.subtract(full, fullcontours)

    # skeletonization
    size = np.size(arrows)
    skel = np.zeros(arrows.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        opened = cv2.morphologyEx(arrows, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(arrows, opened)
        
        eroded = cv2.erode(arrows, element)
        
        skel = cv2.bitwise_or(skel,temp)
        arrows = eroded.copy()
        
        if cv2.countNonZero(arrows)==0:
            break

    arrows = cv2.subtract(full, fullcontours)

    # horizontal lines
    horizontal = np.copy(skel)

    cols = horizontal.shape[1]
    horizontal_size = 15

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # vertical lines
    vertical = np.copy(skel)

    rows = vertical.shape[0]
    verticalsize = 15

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    # making kernels for line filtering
    horizontalKernel = np.zeros((5,5), dtype = np.uint8)
    verticalKernel = horizontalKernel.copy()

    horizontalKernel[2] = [1,1,1,1,1]

    for line in verticalKernel:
        line[2] = 1

    # get rid of intersection pixels
    horizontalTemp = cv2.dilate(horizontal, horizontalKernel)
    verticalTemp = cv2.dilate(vertical, verticalKernel)

    vertical = cv2.subtract(vertical, horizontalTemp)
    horizontal = cv2.subtract(horizontal, verticalTemp)

    # finding the coordinates of the lines
    lines = []

    # horizontal
    contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        x, y, w, h = cv2.boundingRect(approx)
        
        lines.append([(x-1,y),(x+w+1,y)])

    # vertical
    contours, hierarchy = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        x, y, w, h = cv2.boundingRect(approx)
        
        lines.append([(x,y-1),(x,y+h+1)])

    # finding endpoints and intersections
    closeThresh = 30
    endpoints = []
    intersections = []

    lc = 0
    for line in lines:
        ac = 0
        for area in areas:
            if pointCloseToRect(line[0], area[0], closeThresh):
                endpoints.append([line[0], lc, ac, getPointOrientation(line[0],area[0])])
            
            if pointCloseToRect(line[1], area[0], closeThresh):
                endpoints.append([line[1], lc, ac, getPointOrientation(line[1],area[0])])

            ac += 1
        
        IcontainsS = False
        IcontainsE = False
        for i in intersections:
            if (not IcontainsS) and pointsClose(line[0], i[0], closeThresh):
                IcontainsS = True
            if (not IcontainsE) and pointsClose(line[1], i[0], closeThresh):
                IcontainsE = True
        
        intersectCountS = 1
        intersectCountE = 1
        for line2 in lines:
            if line != line2:
                if (not IcontainsS):
                    if (pointsClose(line[0],line2[0],closeThresh)):
                        intersectCountS += 1
                    elif (pointsClose(line[0],line2[1],closeThresh)):
                        intersectCountS += 1
                if (not IcontainsE):
                    if (pointsClose(line[1],line2[0],closeThresh)):
                        intersectCountE += 1
                    elif (pointsClose(line[1],line2[1],closeThresh)):
                        intersectCountE += 1
        
        if intersectCountS > 2:
            intersections.append([line[0], intersectCountS])
        if intersectCountE > 2:
            intersections.append([line[1], intersectCountE])
        
        lc += 1

    # getting every endpoint connected to the intersections
    ISep = [] # intersetion endpoints
    inspectedIS = []
    for i in range(len(intersections)):
        x = 0
        while x < len(inspectedIS) and not inspectedIS[x] == i:
            x += 1
        if x == len(inspectedIS):
            ep, inspectedIS = getIntersectionEndpoints(i, intersections, lines, endpoints, closeThresh, inspectedIS)
            ISep.append(ep)

    # finding every final lines (from shape to shape) except ones that pass intersection
    fullLines = []
    for i in range(len(endpoints)):
        y = 0
        x = 0
        while y < len(ISep) and not ISep[y][x] == endpoints[i][0]:
            x = 0
            while x < len(ISep[y]) and not ISep[y][x] == endpoints[i][0]:
                x += 1
            if x == len(ISep[y]):
                y += 1
                x = 0
        
        if y == len(ISep):
            e = endpoints[i]
            
            if lines[e[1]][0] == e[0]:
                otherSide = lines[e[1]][1]
            else:
                otherSide = lines[e[1]][0]
            
            otherSideLineIdx = e[1]
            
            j = 0
            while j < len(lines):
                if not j == otherSideLineIdx:
                    if pointsClose(lines[j][0], otherSide, closeThresh):
                        otherSide = lines[j][1]
                        otherSideLineIdx = j
                        j = -1
                    elif pointsClose(lines[j][1], otherSide, closeThresh):
                        otherSide = lines[j][0]
                        otherSideLineIdx = j
                        j = -1
                
                j += 1
            
            x = 0
            while x < len(fullLines) and (not fullLines[x][0] == otherSide or not fullLines[x][1] == e[0]):
                x += 1
            
            if x == len(fullLines):
                y = 0
                while y < len(endpoints) and not endpoints[y][0] == otherSide:
                    y += 1
                fullLines.append([e[0], otherSide, 0, e[2], endpoints[y][2], e[3], endpoints[y][3]])

    # finding every arrowhead
    arrowheads = cv2.subtract(255-thresh, fullcontours)
    
    for l in lines:
        cv2.line(arrowheads,l[0],l[1],0,erodeCount*2)

    # figuring out the direction of the arrow
    inspectArea = 12

    arrowheads = cv2.threshold(arrowheads, 240, 255, cv2.THRESH_BINARY)[1]

    # simple lines
    for l in fullLines:
        mainArea = arrowheads[l[0][1]-inspectArea:l[0][1]+inspectArea, l[0][0]-inspectArea:l[0][0]+inspectArea]
        otherArea = arrowheads[l[1][1]-inspectArea:l[1][1]+inspectArea, l[1][0]-inspectArea:l[1][0]+inspectArea]
        
        if (cv2.countNonZero(otherArea) > cv2.countNonZero(mainArea)):
            l[2] = 1

    # intersection lines
    for i in range(len(ISep)):
        ISArrowPixelCount = []
        ISArrowStarts = []
        ISArrowEnds = []
        
        avg = 0
        for j in range(len(ISep[i])):
            pt = ISep[i][j]
            
            mainArea = arrowheads[pt[1]-inspectArea:pt[1]+inspectArea, pt[0]-inspectArea:pt[0]+inspectArea]
            
            ISArrowPixelCount.append([j, cv2.countNonZero(mainArea)])
            
            avg += ISArrowPixelCount[-1][1]
        
        avg = avg / len(ISArrowPixelCount)
        
        for isa in ISArrowPixelCount:
            if isa[1] < avg:
                ISArrowStarts.append(isa[0])
            else:
                ISArrowEnds.append(isa[0])
        
        for x in range(len(ISArrowStarts)):
            for y in range(len(ISArrowEnds)):
                s = ISArrowStarts[x]
                e = ISArrowEnds[y]
                
                es = 0
                while es < len(endpoints) and not endpoints[es][0] == ISep[i][s]:
                    es += 1
                ee = 0
                while ee < len(endpoints) and not endpoints[ee][0] == ISep[i][e]:
                    ee += 1
                
                fullLines.append([ISep[i][s], ISep[i][e], 1, endpoints[es][2], endpoints[ee][2], endpoints[es][3], endpoints[ee][3]])
    
    isArrowdetectionSuccessful = True
except:
    isArrowdetectionSuccessful = False
    print("Arrow detection failed")

# drawing elements to the output window
for area in areas:
    cv2.rectangle(img,(area[0][0],area[0][1]),(area[0][0]+area[0][2], area[0][1]+area[0][3]),(0,255,0),2)

if isArrowdetectionSuccessful:
    for l in fullLines:
        if l[2] == 0:
            cv2.arrowedLine(img, (l[1]), (l[0]), (255,0,255),3)
        else:
            cv2.arrowedLine(img, (l[0]), (l[1]), (255,0,255),3)

    for e in endpoints:
        cv2.rectangle(img,(e[0][0]-2,e[0][1]-2),(e[0][0]+4, e[0][1]+4),(0,255,0),2)

    for e in intersections:
        cv2.rectangle(img,(e[0][0]-5,e[0][1]-5),(e[0][0]+10, e[0][1]+10),(255,0,0),3)

    for l in lines:
        cv2.line(img, (l[0]), (l[1]), (0,0,255),1)


# creating xml file (output.xml) for draw.io
file = open('output.xml','w')

file.write('<mxGraphModel dx="200" dy="200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="400" pageHeight="400" math="0" shadow="0">\n<root>\n')

file.write('<mxCell id="0" />\n<mxCell id="1" parent="0" />\n')

id = 2

# shapes
for area in areas:
    if area[1] == 'T':
        # triangle
        file.write('<mxCell id="'+str(id)+'" value="" style="triangle;whiteSpace=wrap;html=1;" vertex="1" parent="1">\n<mxGeometry x="'+str(area[0][0])+'" y="'+str(area[0][1])+'" width="'+str(area[0][2])+'" height="'+str(area[0][3])+'" as="geometry" />\n</mxCell>\n')
    elif area[1] == 'R':
        # rectangle
        file.write('<mxCell id="'+str(id)+'" value="" style="rounded=0;whiteSpace=wrap;html=1;" vertex="1" parent="1">\n<mxGeometry x="'+str(area[0][0])+'" y="'+str(area[0][1])+'" width="'+str(area[0][2])+'" height="'+str(area[0][3])+'" as="geometry" />\n</mxCell>\n')
    elif area[1] == 'D':
        # diamond
        file.write('<mxCell id="'+str(id)+'" value="" style="rhombus;whiteSpace=wrap;html=1;" vertex="1" parent="1">\n<mxGeometry x="'+str(area[0][0])+'" y="'+str(area[0][1])+'" width="'+str(area[0][2])+'" height="'+str(area[0][3])+'" as="geometry" />\n</mxCell>\n')
    elif area[1] == 'P':
        # parallelogram
        file.write('<mxCell id="'+str(id)+'" value="" style="shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;" vertex="1" parent="1">\n<mxGeometry x="'+str(area[0][0])+'" y="'+str(area[0][1])+'" width="'+str(area[0][2])+'" height="'+str(area[0][3])+'" as="geometry" />\n</mxCell>\n')
    else:
        # elipse
        file.write('<mxCell id="'+str(id)+'" value="" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">\n<mxGeometry x="'+str(area[0][0])+'" y="'+str(area[0][1])+'" width="'+str(area[0][2])+'" height="'+str(area[0][3])+'" as="geometry" />\n</mxCell>\n')

    id += 1

# lines
if isArrowdetectionSuccessful:
    for l in fullLines:
        if l[2] == 0:
            exitX, exitY = getOrientationValues(l[6])
            entryX, entryY = getOrientationValues(l[5])
            file.write('<mxCell id="'+str(id)+'" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX='+str(exitX)+';exitY='+str(exitY)+';exitDx=0;exitDy=0;entryX='+str(entryX)+';entryY='+str(entryY)+';entryDx=0;entryDy=0;" edge="1" parent="1" source="'+str(l[4]+2)+'" target="'+str(l[3]+2)+'">\n<mxGeometry relative="1" as="geometry" />\n</mxCell>\n')
        else:
            exitX, exitY = getOrientationValues(l[5])
            entryX, entryY = getOrientationValues(l[6])
            file.write('<mxCell id="'+str(id)+'" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX='+str(exitX)+';exitY='+str(exitY)+';exitDx=0;exitDy=0;entryX='+str(entryX)+';entryY='+str(entryY)+';entryDx=0;entryDy=0;" edge="1" parent="1" source="'+str(l[3]+2)+'" target="'+str(l[4]+2)+'">\n<mxGeometry relative="1" as="geometry" />\n</mxCell>\n')
        
        id += 1

file.write('</root>\n</mxGraphModel>\n')

file.close()

cv2.imshow("Processed image", img)

print("DONE")

cv2.waitKey()
cv2.destroyAllWindows()