import math
import numpy as np
from modules.GTT_utils import calcDistance, calcDistancePoints

### Constants ###
STRIPE_SIZE_IN_PERCENT = 5

'''
Internal class that represnts a stripe in the image
'''
class _Stripe:
    def __init__(self,pointAmount,place,points):
        self._pointAmount = pointAmount
        self._place = place
        self._points = points

### Functions ###

'''
Counts the relevant number of points between the relevant values.
'''
def _getPointAmount(pointList, minV, maxV, func):
    amount = 0
    tempList = []
    for x, y in pointList:
        if func(x, y) >= minV and func(x, y) <= maxV:
            flag = True
            for x1, y1 in tempList:
                if calcDistance(x, y, x1, y1) < 5:
                    flag = False
                    break
            if flag:
                amount = amount + 1
                tempList.append((x, y))
    return amount

'''
Returns all the points at pointList inside the given x,y values.
'''
def _getAllPointsInRange(pointList, minYmaxY, func):
    minY = minYmaxY[0]
    maxY = minYmaxY[1]
    retList = []
    for x, y in pointList:
        if func(x, y) >= minY and func(x, y) <= maxY:
            retList.append((x, y))
    return retList

'''
Find the vertical/horizontal stripe with most dots and returns all the points in it.
The direction of the stripe is defined according to the function pointer given as 'func'.
'''
def _findStripes(pointList, width, height, func):
    stepSize = float(height * STRIPE_SIZE_IN_PERCENT / 100) # Take STRIPE_SIZE_IN_PERCENT percent
    minValue = 0
    maxValue = stepSize
    stripes = []
    while True:
        pointAmount = _getPointAmount(pointList, minValue, maxValue, func)
        stripePoints = _getAllPointsInRange(pointList, (minValue, maxValue), func)
        stripes.append(_Stripe(pointAmount,(minValue, maxValue),stripePoints))

        minValue = minValue + stepSize / 4
        maxValue = maxValue + stepSize / 4
        if minValue >= width:
            break
        if maxValue > width:
            maxValue = width

    return stripes

'''
Using the axes received - finds the intersection points and returns their median as origin point.
'''
def _findOrig(res, x_axis, y_axis):
    suspectedOrig = [pt for pt in res if pt in x_axis and pt in y_axis]
    if len(suspectedOrig) == 0:
        return None

    orig = [0, 0]
    for pt in suspectedOrig:
        orig[0] = orig[0] + pt[0]
        orig[1] = orig[1] + pt[1]
    orig = np.array([orig[0] / len(suspectedOrig), orig[1] / len(suspectedOrig)])
    return orig

'''
Returns true if pt1 is closest to pt inside container
All points are two dimensional
'''
def _closestPoints(pt1,container,pt):
        flag = True
        for pt2 in container:
            if math.fabs(pt2[0] - pt1[0]) > 30 or (pt2[0] == pt1[0] and pt2[1] == pt1[1]):
                continue
            if math.fabs(pt2[1] - pt[1]) < math.fabs(pt1[1] - pt[1]):
              flag = False
              break
        
        return flag

'''
Interface function - finds axes and origin points and returns them.
'''
def getGraphAxesAndOrig(pointList, height, width):
    y_axis_cand = _findStripes(pointList, width, height, (lambda x, y: x))
    x_axis_cand = _findStripes(pointList, height, width, (lambda x, y: y))
    
    x_axis = sorted(x_axis_cand,key = lambda cand: cand._pointAmount)[-1]._points
    max_x = sorted(x_axis,key = lambda val: val[0])[-1] 

    y_axis_sorted = sorted(y_axis_cand,key = lambda cand: cand._place[0])

    y_filtered = []
    orig = None
    for cand in y_axis_sorted:
        tmpOrig = _findOrig(pointList,x_axis,cand._points)
        if not tmpOrig is None:
            y_filtered.append(cand)
            if orig is None:
                orig = tmpOrig
            elif calcDistancePoints(orig,tmpOrig) > 100:
                break
    
    y_axis = sorted(y_filtered,key=lambda s: s._pointAmount)[-1]._points

    filtered_x_axis = [pt for pt in x_axis if _closestPoints(pt,x_axis,orig)]
    filtered_x_axis.append(max_x)
    return filtered_x_axis, y_axis, orig
