"""
Given two 2D polygons write a function that calculates the IoU of their areas,
defined as the area of their intersection divided by the area of their union.
The vertices of the polygons are constrained to lie on the unit circle and you
can assume that each polygon has at least 3 vertices, given and in sorted order.

- You are free to use basic math functions/libraries (sin, cos, atan2, numpy etc)
  but not geometry-specific libraries (such as shapely).
- You are free to look up geometry-related formulas, optionally copy paste in
  short code snippets and adapt them to your needs.
- We do care and evaluate your general code quality, structure and readability
  but you do not have to go crazy on docstrings.
"""
import numpy as np
from math import atan2, sqrt

def computeLine(p1, p2):
    
    """
    Computing the lines parameters m and b given two points
    
    """

    
    dx = p1[0] - p2[0]
    return (p1[1] - p2[1]) / dx, (p1[0] * p2[1] - p2[0] * p1[1]) / dx
            

def cleanPoly(poly):
    
    """
    Removing duplicate points from polygon list of vertices
    
    """

    
    cleanedPoly = []
    for point in poly:
        if isNotInList(point, cleanedPoly):
            cleanedPoly.append(point)
            
    return cleanedPoly
    
    
    
def computeIntersection(p11, p12, p21, p22, tol=1e-6):
    
    """
    Computing intersection of two lines
    
    """
    
    # Compute difference in x co-ordinates for two lines
    
    dx1, dx2 = p11[0] - p12[0], p21[0] - p22[0]
    
    # If both difference in x are below tolerance, the lines are vertical and parallel => no intersection
            
    if abs(dx1) < tol and abs(dx2) < tol:
        return None
    
    # If just the first line difference in x is below tolerance, the first line is vertical
    elif abs(dx1) < tol:
            x = (p11[0] + p12[0])/2 # x co-ordinate of intersection
            m2, b2 = computeLine(p21, p22) # get second line parameters
            return x, m2 * x + b2
    
    # If just the second line difference in x is below tolerance, the second line is vertical
    elif abs(dx2)< tol:
            x = (p21[0] + p22[0]) / 2 # x co-ordinate of intersection
            m1, b1 = computeLine(p11, p12) # get first line parameters
            return x, m1 * x + b1 
    
    # If none of the differences in x difference is below tolerance, none of the line is vertical
    
    else:
            m1,b1 = computeLine(p11, p12) # get first line parameters
            m2,b2 = computeLine(p21, p22) # get second line parameters
            dm = m1 - m2 # difference in slope
            
            # if difference in slope is below tolerance, the two lines are parallel i.e. no intersection
            if abs(dm) < tol:
                return None
            # else, compute intersection x, y
            else:
                return (b2 - b1) / dm, (m1 * b2 - b1 * m2) / dm
            
def distanceBetween(p1, p2):
    
    """
    Euclidean distance between two points
    
    """
        
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            
            
            
def iou(poly1, poly2):
    
    """
    Computing intersection over union for given polygon
    
    """
    
    poly1, poly2 = cleanPoly(poly1), cleanPoly(poly2) # clean polygons of duplicate points
    poly3 = polyIntersection(poly1, poly2) # compute intersected polygon      
    
    # if intersection exists (its a convex polygon with atleat 3 vertices)
    
    if poly3:
        # converting polygons to np arrays
        
        poly1 = np.array(poly1, dtype = np.float32)
        poly2 = np.array(poly2, dtype = np.float32)
        poly3 = np.array(poly3, dtype = np.float32)
        
        # computing area of intersection 
        intersectionArea = polyArea(poly3[:,0], poly3[:,1])
        
        # computing area of union (= area of polygon - area of intersection)   
        unionArea = polyArea(poly1[:,0], poly1[:,1]) + polyArea(poly2[:,0], poly2[:,1]) - intersectionArea
        
        
        # IoU = area of intersection / area of union
        return intersectionArea / unionArea
    
    # else, ploygons do not intersect i.e. IoU = 0.0
    else:
           
        return 0.0

    
def isNotInList(point, list, tol=1e-6):
    
    """
    Checking if the point is already in the list
    
    """
    
    for p in list:
        
        if distanceBetween(point, p) < tol:
            return False
    
    return True
    
    
    
def polyArea(x, y):
    
    """
    Computing the area of polygon using Shoeloace formula given ordered x and y coordinates of its vertices. (https://en.wikipedia.org/wiki/Shoelace_formula).
    
    """
    
    area = 0.5 * np.abs(np.dot(y, np.roll(x,1)) - np.dot(x, np.roll(y,1)))
    return area 
            
def polyIntersection(poly1, poly2):
    
    """
    Computing the intersection between two polygons
    
    """
    
    intersections, orientations = [], [] # list of intersection points and respective orientations w.r.t the origin
    n1, n2 = len(poly1), len(poly2) # no. of vertices of two polygons
    
    # for each vertex in 1st polygon
    for i, currentVertex1 in enumerate(poly1):
        

        previousVertex1 = poly1[(i + n1 - 1) % n1]         # pervious vertex of 1st polygon
        
        # bounding box of current edge of 1st polygon
        xMax = max(currentVertex1[0], previousVertex1[0])
        xMin = min(currentVertex1[0], previousVertex1[0])
        yMax = max(currentVertex1[1], previousVertex1[1])
        yMin = min(currentVertex1[1], previousVertex1[1])
        
        # for each vertex in 2nd polygon    
        for j, currentVertex2 in enumerate(poly2):
            
            previousVertex2 = poly2[(j + n2 - 1) % n2]
            
            # compute intersection between 2 lines of 2 polygons
            intersect = computeIntersection(currentVertex1, previousVertex1, currentVertex2, previousVertex2)
            
            # if intersection exists, it is in the bounding box and has not been already accounted
            if intersect:
            
                if xMin <= intersect[0] <= xMax and yMin <= intersect[1] <= yMax:
                    if isNotInList(intersect, intersections):
                        
                        intersections.append(intersect) # appending it to the list
                        orientations.append(atan2(intersect[1], intersect[0])) # appending to the corresponding orientation
    
    # if fewer than 3 vertices
    if len(intersections) < 3:
        
        return None # its not a polygon (intersection is null)
    else:
        
        #sorting the vertices of polygon by orientation
        intesectionPoly = [x for _, x in sorted(zip(orientations, intersections))]
        return intesectionPoly

# --------------------------------------------------------

if __name__ == "__main__":

    cases = []
    # Case 1: a vanilla case (see https://imgur.com/a/dSKXHPF for a diagram)
    poly1 = [
        (-0.7071067811865475, 0.7071067811865476),
        (0.30901699437494723, -0.9510565162951536),
        (0.5877852522924729, -0.8090169943749476),
    ]
    poly2 = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (0.7071067811865475, -0.7071067811865477),
    ]
    cases.append((poly1, poly2, "simple case", 0.12421351279682288))
    # Case 2: another simple case
    poly1 = [
        (1, 0),
        (0, 1),
        (-0.7071067811865476, -0.7071067811865476),
    ]
    poly2 = [
        (-0.1736481776669303, 0.984807753012208),
        (-1, 0),
        (0, -1),
    ]
    cases.append((poly1, poly2, "simple case 2", 0.1881047657147776))
    # Case 3: yet another simple case, note the duplicated point
    poly1 = [
        (0, -1),
        (-1, 0),
        (-1, 0),
        (0, 1),
    ]
    poly2 = [
        (0.7071067811865476, 0.7071067811865476),
        (-0.7071067811865476, 0.7071067811865476),
        (-0.7071067811865476, -0.7071067811865476),
        (0.7071067811865476, -0.7071067811865476),
        (0.7071067811865476, -0.7071067811865476),
    ]
    cases.append((poly1, poly2, "simple case 3", 0.38148713966109243))

    # Case 4: shared edge
    poly1 = [
        (-1, 0),
        (-0.7071067811865476, -0.7071067811865476),
        (0.7071067811865476, -0.7071067811865476),
        (1, 0),
    ]
    poly2 = [
        (0, 1),
        (-1, 0),
        (1, 0),
    ]
    cases.append((poly1, poly2, "shared edge", 0.0))

    # Case 5: same polygon
    poly1 = [
        (0, -1),
        (-1, 0),
        (1, 0),
    ]
    poly2 = [
        (0, -1),
        (-1, 0),
        (1, 0),
    ]
    cases.append((poly1, poly2, "same same", 1.0))

    # Case 6: polygons do not intersect
    poly1 = [
        (-0.7071067811865476, 0.7071067811865476),
        (-1, 0),
        (-0.7071067811865476, -0.7071067811865476),
    ]
    poly2 = [
        (0.7071067811865476, 0.7071067811865476),
        (1, 0),
        (0.7071067811865476, -0.7071067811865476),
    ]
    cases.append((poly1, poly2, "no intersection", 0.0))


    import time
    t0 = time.time()

    for poly1, poly2, description, expected in cases:
        computed = iou(poly1, poly2)
        print('-'*20)
        print(description)
        print("computed:", computed)
        print("expected:", expected)
        print("PASS" if abs(computed - expected) < 1e-8 else "FAIL")

    # details here don't matter too much, but this shouldn't be seconds
    dt = (time.time() - t0) * 1000
    print("done in %.4fms" % dt)
