# ==========================================================================================================================
# WARP AFFINE

# cv2.circle(img, (710,313), 10, (255,0,255), -1)
# cv2.circle(img, (1770, 312), 10, (255,0,255), -1)
# cv2.circle(img, (700, 2180), 10, (255,0,255), -1)
# cv2.circle(img, (1750, 2200), 10, (255,0,255), -1)

# warp_pt1 = np.float32([[710,313], [1770, 312], [680, 2180], [1750, 2200]])
# height, width = 2154, 1060
# warp_pt2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
# mtx = cv2.getPerspectiveTransform(warp_pt1, warp_pt2)
# output = cv2.warpPerspective(img, mtx, (width, height))

# cv2.imshow("warp", cv2.resize(output, (int(output.shape[1]*PICTURE_SCALE/100), int(output.shape[0]*PICTURE_SCALE/100))))


# ==========================================================================================================================
# GET LINES INTERSECTION

# def getLinesIntersection(l1: Line, l2: Line) -> Point:
#     """Get the intersecion of 2 lines (infinite lines)."""
#     x1 = l1.p1.x
#     x2 = l1.p2.x
#     y1 = l1.p1.y
#     y2 = l1.p2.y

#     x3 = l2.p1.x
#     x4 = l2.p2.x
#     y3 = l2.p1.y
#     y4 = l2.p2.y

#     D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)

#     return Point(round(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/D), round(((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/D))


# ==========================================================================================================================
# FANCY TODO

# TODO:  ######################################################################
# TODO:  #                                                                    #
# TODO:  #  TEST KERAS WITH REGULAR INTERSECTION AND CONNECTED INTERSECTION   #
# TODO:  #                                                                    #
# TODO:  ######################################################################


# ==========================================================================================================================
# CIRCLE DETECTION

# rows = gray.shape[0]
#     circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
#                                param1=100, param2=30,
#                                minRadius=1, maxRadius=30)
    
    
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             center = (i[0], i[1])
#             # circle center
#             cv.circle(src, center, 1, (0, 100, 100), 3)
#             # circle outline
#             radius = i[2]
#             cv.circle(src, center, radius, (255, 0, 255), 3)

# ==========================================================================================================================
# FIND INTERSECTIONS

# intersectCountS = 1
# intersectCountE = 1
# for line2 in lines:
#     if line != line2:
#         if (not IcontainsS):
#             if (pointsClose(line[0],line2[0],closeThresh)):
#                 intersectCountS += 1
#             elif (pointsClose(line[0],line2[1],closeThresh)):
#                 intersectCountS += 1
#         if (not IcontainsE):
#             if (pointsClose(line[1],line2[0],closeThresh)):
#                 intersectCountE += 1
#             elif (pointsClose(line[1],line2[1],closeThresh)):
#                 intersectCountE += 1

# if intersectCountS > 2:
#     intersections.append([line[0], intersectCountS])
# if intersectCountE > 2:
#     intersections.append([line[1], intersectCountE])


# ==========================================================================================================================
# SKELETONIZATION

# temptresh = thresh.copy()

# size = np.size(temptresh)
# skel = np.zeros(temptresh.shape, np.uint8)

# # Get a Cross Shaped Kernel
# element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

# # Repeat steps 2-4
# while True:
#     #Step 2: Open the image
#     open = cv2.morphologyEx(temptresh, cv2.MORPH_OPEN, element)
#     #Step 3: Substract open from the original image
#     temp = cv2.subtract(temptresh, open)
#     #Step 4: Erode the original image and refine the skeleton
#     eroded = cv2.erode(temptresh, element)
#     skel = cv2.bitwise_or(skel,temp)
#     temptresh = eroded.copy()
#     # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
#     if cv2.countNonZero(temptresh)==0:
#         break

# # Displaying the final skeleton
# cv2.imshow("Skeleton",resizeImage(skel, PICTURE_SCALE))