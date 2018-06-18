import cv2
import math
from PIL import Image, ImageEnhance, ImageFilter
import keras
import copy
import numpy
import tensorflow as tf

# Global variable for log level. It will be initialized in the recgonize function
_LOG_LEVEL = 0

# Internal imports
from modules.GTT_utils import getEmptyImageName
from modules.GTT_utils import createFolder
from modules.GTT_utils import calcDistancePoints
from modules.GTT_utils import deleteFolder
from modules.GTT_utils import appendToFile
from modules.GTT_utils import deleteAllFilesByTypeInFolder
from modules.GTT_utils import checkIfPathExists

### Constants ###
# Minimum threshold for detecting image
MINIMUM_PRECENT_FOR_DETECTING_DIGIT_IN_IMAGE = 0.5
FILTER_POINTS_ON_AXIS = 10
# Epsilin rec value should be calculated with the image size
EPSILON_REC = 3 # Old value: 50
DIGIT_RADIUS_FACTOR = 7
LOG_FILE = "/log.txt"

### Functions ###
def calcRadiusForDigit(h, w):
    return (min(h,w) / DIGIT_RADIUS_FACTOR)

'''
Helper class. This class holds digits that should represent one number in the graph
'''
class _Number():
    def __init__(self, imageHeight, imageWidth, digit, pt1, pt2):
        self.digits = [(digit, (pt1, pt2))]
        self.maxX = pt1
        self.maxY = pt2
        self.imageHeight = int(imageHeight)
        self.imageWidth = int (imageWidth)

    def addDigit(self, digit, pt1, pt2):
        flag = False
        threshold = calcRadiusForDigit(self.imageHeight, self.imageWidth)
        for _, (p1, p2) in self.digits:
            if math.fabs(pt1 - p1) < threshold and math.fabs(pt2 - p2) < threshold:
                flag = True
                if pt1 > self.maxX:
                    self.maxX = pt1
                if pt2 > self.maxY:
                    self.maxY = pt2
        if flag:
            self.digits.append((digit, (pt1, pt2)))
        return flag
    def __str__(self):
        return ''.join("D: " + str(d) + ", p1: " + str(p1) + ",p2: " + str(p2) + "\n" for d, (p1, p2) in self.digits)
    def getNumber(self):
        return ''.join(str(d) for d, _ in sorted(self.digits, key = lambda x : x[1][1]))

'''
Helper class. This class holds a rectangle from the graph. Rectangle should be a digit
'''
class _TemporaryRectangle():
    def __init__(self, pt1, pt2, xyStart, xyEnd, imagePath, recSize, index):
        self.pt1 = pt1
        self.pt2 = pt2
        self.xyStart = xyStart
        self.xyEnd = xyEnd
        self.imagePath = imagePath
        self.recSize = recSize
        self.index = index
    def getPt1AndPt2(self):
        return (self.pt1, self.pt2)        
    def getImagePath(self):
        return self.imagePath
    def getXyStartXyEnd(self):
        return (self.xyStart, self.xyEnd)
    def getRectSize(self):
        return self.recSize
    def getAllRecCoordinates(self):
        xy_left_bottom = self.xyStart
        xy_right_top = self.xyEnd
        xy_left_top = (self.xyStart[0], self.xyEnd[1])
        xy_right_bottom = (self.xyEnd[0], self.xyStart[1])
        return [xy_left_bottom, xy_left_top, xy_right_top, xy_right_bottom]
    def getIndex(self):
        return self.index

def _printAxis(x_axis, y_axis, h, w):
    if x_axis is None and y_axis is None:
        return

    import matplotlib.pyplot as plt

    x_list = [x for x, y in x_axis] + [x for x, y in y_axis]
    y_list = [y for x, y in x_axis] + [y for x, y in y_axis]
    plt.scatter(x_list, y_list)
    plt.axis([0, w, 0, h])
    plt.title('Axis')
    plt.show()

'''
The function get a list of digits, and collect it to a list of numbers
'''
def _collectDigitsInGroups(unorderedPointsList, imageHeight, imageWidth):
    numbersList = [] 
    i = -1
    pointList = sorted(unorderedPointsList, key=(lambda x: x[1][1]), reverse=True)

    for digit, (pt1, pt2) in pointList:
        i = i + 1
        tempNumber = _Number(imageHeight, imageWidth, digit, pt1, pt2)
        if digit == 'X':
            continue
        j = -1
        for digit2, (pt3, pt4) in pointList:
            j = j + 1
            if pt1 == pt3 and pt2 == pt4:
                continue
            if tempNumber.addDigit(digit2, pt3, pt4):
                pointList[j] = ('X', (100000, 100000))
        
        numbersList.append(tempNumber)
        pointList[i] = ('X', (100000, 100000))
    if _LOG_LEVEL >= 4:
        print ("Detected numbers")
        for x in numbersList:
            print (str(x))
    return [numbersList[0], numbersList[-1]]

'''
The function create a new image with a padding of zeros
'''
def _createImg(img, h, w):
    # Calc padding
    originalWidth, originalHeight = img.shape
    height_diff = int((h - originalHeight) / 2)
    if (h - originalHeight) % 2 == 1:
        height_diff = height_diff + 1
    width_diff = int((w - originalWidth) / 2)
    if (w - originalWidth) % 2 == 1:
        width_diff = width_diff + 1

    # Pad the original image + normalize
    north = height_diff
    south = h - originalHeight - north
    west = width_diff
    east = w - originalWidth - west
    for i in range(originalWidth):
        for j in range(originalHeight):
            if img[i][j] < 127:
                img[i][j] = 1
            else:
                img[i][j] = 0
    retValue = numpy.pad(img, ((west, east), (north, south)), 'constant')
    return retValue

'''
The function try to detect the digit in the number
'''
def _recognizeWithKeras(modelName, imagePath, saveImageAfterResize, folderName, imageIndex):
    # Load model
    model = keras.models.load_model(modelName)
    
    # Resize image - save ratio
    size = 20, 20
    img = Image.open(imagePath)
    img.thumbnail(size, Image.ANTIALIAS)

    # Load image in openCV format
    tempImagePath = getEmptyImageName(folderName, 'filtered' + str(imageIndex), "png")
    img.save(tempImagePath, "png")
    img = cv2.imread(tempImagePath, 0)
    
    # Pad + normalization
    img = _createImg(img, 28, 28)
    if saveImageAfterResize:
        appendToFile(folderName + LOG_FILE, "Image source:" + tempImagePath + "\n")
        appendToFile(folderName + LOG_FILE, str(img))
        appendToFile(folderName + LOG_FILE, "\n")
    
    img = img.reshape(1, 28,28,-1) # For 12
    ans = model.predict(img)

    if saveImageAfterResize:
        for i in range(len(ans[0])):
            floatNumer = ("%0.2f" % (ans[0][i] * 100))
            appendToFile (folderName + LOG_FILE, str(i) + ": " + str(floatNumer) + "%\n")
        appendToFile(folderName + LOG_FILE, "##########################################\n")
    return ans[0]

'''
The function receieves a model and a source file - runs the source file on the model and returns the output.
Returns -1 in case that it couldn't determine the digit in the image.
'''
def _detectNumber(modelName, source, folderName, imageIndex):
    predictionList = _recognizeWithKeras(modelName, source, True, folderName, imageIndex)
    suspectedDigit = 0
    for i in range(len(predictionList)):
        if predictionList[i] > predictionList[suspectedDigit]:
            suspectedDigit = i
    if predictionList[suspectedDigit] < MINIMUM_PRECENT_FOR_DETECTING_DIGIT_IN_IMAGE:
        return -1
    return suspectedDigit

'''
Checks if the given point is inside the rect.
Rect list positions:
  1   2
  0   3
'''
def _isPointInsideRect(point, rect, epsilon):
    return point[0] >= (rect[0][0] - epsilon) and \
           point[0] <= (rect[1][0] + epsilon) and \
           point[1] >= (rect[0][1] - epsilon) and \
           point[1] <= (rect[1][1] + epsilon)

'''
The function try to filter rectangles that don't represent a digit
'''
def _maybeFilterRectangle(filteredRect, rectList, imageSize, heigth, width):
    if filteredRect.getRectSize() < imageSize * 0.00001:
        return True
    for tRec in rectList:
        # filteredRect == tRec
        if calcDistancePoints(tRec.getXyStartXyEnd()[0], filteredRect.getXyStartXyEnd()[0]) == 0:
            continue
        
        tRec_list = tRec.getAllRecCoordinates()
        filteredRect_list = filteredRect.getAllRecCoordinates()
        flag = False
        for p in filteredRect_list:
            factor = (min (heigth, width)) / 200
            if _isPointInsideRect(p, tRec_list,factor):
                flag = True
                break

        if flag and tRec.getRectSize() > filteredRect.getRectSize():
            return True
    return False

'''
the function check if the given rectange is placed below the axis (below X axis and in the left
side of the Y axis)
'''
def _isRectBelowAxis(xyStart, xyEnd, xMaxInYAxis, yMaxInXAxis):
    if xMaxInYAxis == -1:
        return False

    xMaxInYAxis = xMaxInYAxis - 10
    yMaxInXAxis = yMaxInXAxis - 10
    if not (xyStart[0] > xMaxInYAxis or xyEnd[0] > xMaxInYAxis):
        return True

    if xyStart[1] > yMaxInXAxis or xyEnd[1] > yMaxInXAxis:
        return True

    return False

'''
the function sorted the list and return the i element
'''
def _getMaxValueFromAxis(l, d, i):
    return sorted(l, key=d)[i]
'''
random rgb (0-255, 0-255, 0-255)
'''
def _getRandomRGB():
    from random import randint
    ra = (randint(0, 255), randint(0, 255), randint(0, 255))
    return ra

'''
this function get a list of points and two points that represnts a rectanges.
return- true if the recatngle placed near the points in the list
'''
def _isRectOnAxisAux(l, p1, p2):
    startFlag = endFlag = False
    for x,y in l:
        if calcDistancePoints(p1, (x,y)) < FILTER_POINTS_ON_AXIS:
            startFlag = True
        if calcDistancePoints(p2, (x, y)) < FILTER_POINTS_ON_AXIS:
            endFlag = True
        if startFlag and endFlag:
            return True
    return False
'''
this function get a list of points and two points that represnts a rectanges.
return- true if the recatngle placed near the points in the list
'''
def _isRectOnAxis(x_axis, y_axis, xyStart, xyEnd):
    if x_axis is None or y_axis is None:
        return False
    xyStartSecond = (xyStart[0], xyEnd[1])
    xyEndSecond = (xyEnd[0], xyStart[1])
    return _isRectOnAxisAux(x_axis, xyStart, xyEnd) or _isRectOnAxisAux(y_axis, xyStart, xyEnd) or \
        _isRectOnAxisAux(x_axis, xyStartSecond, xyEndSecond) or _isRectOnAxisAux(y_axis, xyStartSecond, xyEndSecond)
          

'''
The function search for digits in the given image
'''
def _searchDigitsInImage(im, im_copy, point_list, modelName, imageSize, folderName, x_axis, y_axis):
    height, width, _ = im.shape
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    _,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, calculate HOG features and predict
    myIndex = 0
    pointList = []
    recList = []
    
    if y_axis is None or x_axis is None:
        xMaxInYAxis = yMaxInXAxis = -1
    else:
        xMaxInYAxis = _getMaxValueFromAxis(y_axis, lambda p: p[0], -1)[0]
        yMaxInXAxis = _getMaxValueFromAxis(x_axis, lambda p: p[1], 0)[1]

        if _LOG_LEVEL >= 3:
            cv2.line(im, (xMaxInYAxis, 0), (xMaxInYAxis, height), (0, 255, 0), thickness =10)
            cv2.line(im, (0, yMaxInXAxis), (width, yMaxInXAxis), (0, 255, 255), thickness = 10)

    for rect in rects:
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        if pt1 < 0:
            pt1 = 0
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        if pt2 < 0:
            pt2 = 0
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        
        if len(roi) == 0:
            continue
        if roi is None:
            continue
        try:
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        except:
            continue

        # rect[2] - width
        # rect[3] - heigth
        xyStart = (rect[0], rect[1])
        xyEnd = (rect[0] + rect[2], rect[1] + rect[3])
        area = rect[3] * rect[2]

        # Ignore short and thick rects
        if rect[2] > rect[3] * 2:
            continue
        # Filter by maximum rect size
        if imageSize * 0.01 < area:
            continue
        if 0 in rect:
            continue
        # Filter by axis
        if not _isRectBelowAxis(xyStart, xyEnd, xMaxInYAxis, yMaxInXAxis):
            continue
        if _isRectOnAxis(x_axis, y_axis, xyStart, xyEnd):
            continue

        crop_img = im_copy[rect[1] - EPSILON_REC: rect[1] + rect[3] + EPSILON_REC, rect[0] - EPSILON_REC: rect[0] + rect[2] + EPSILON_REC]
        newImagePath = getEmptyImageName(folderName, 'non_filtered', "png", True)
        try:
            cv2.imwrite(newImagePath, crop_img)
        except:
            appendToFile (folderName + LOG_FILE, "Can't create new image")
            continue
                
        recList.append(_TemporaryRectangle(pt1, pt2, xyStart, xyEnd, newImagePath, area, myIndex))

        if _LOG_LEVEL >= 3:
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
            cv2.putText(im, str(myIndex), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

            xCenter = (xyStart[0] + xyEnd[0])/2
            yCenter = (xyStart[1] + xyEnd[1])/2
            cv2.circle(im, (int(xCenter), int(yCenter)), int(calcRadiusForDigit(height, width)), _getRandomRGB(), thickness=10)

        myIndex = myIndex + 1
    
    if _LOG_LEVEL >= 3:
        for x,y in x_axis:
            cv2.circle(im, (x,y), 5, (247, 0, 255))
        for x,y in y_axis:
            cv2.circle(im, (x,y), 5, (37, 145, 199))

    # Filter rectangles
    recListReturn = []
    for tempRec in recList:
        recListReturn.append(tempRec.getXyStartXyEnd())
        if _maybeFilterRectangle(tempRec, recList, imageSize, height, width):
            continue
        digit = _detectNumber(modelName, tempRec.getImagePath(), folderName, tempRec.getIndex())
        if digit == -1:
            continue
        pointList.append((digit, tempRec.getPt1AndPt2()))

    if _LOG_LEVEL >= 3:
        # To add (0, 0) to images rectangles uncomment the next line -
        # cv2.putText(im, str(0), (0, 10),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        _showImageWithDigits(im)

    return pointList, recListReturn

'''
Debug function - shows the rect digits on image.
'''
def _showImageWithDigits(im):
    cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    cv2.waitKey()

'''
The function choose which numbers belong to X-axis and which to Y-axis
'''
def _findXAndYValues(numbersList):
    # Find max X
    xIndex = yMax = 0
    for i, _ in enumerate(numbersList):
        if numbersList[i].maxX > numbersList[xIndex].maxX:
            xIndex = i

    # Find max Y
    yMax = numbersList[1 - xIndex].getNumber()

    return [int(numbersList[xIndex].getNumber()), int(yMax)]

'''
The function get an image path, and return the number in it.
Returns a list:
    index 0 -> X max
    index 1 -> Y max
In case of error - return value will be [-1, -1]
logLevel:
 0 - no logs
 1 - only logs
 2 - logs + images
 3 - logs + images + plt
'''
def recognize(imagePath, point_list = None, x_axis = None, y_axis = None, logLevel = 0):
    modelName = 'modules/keras_12_model.pl'
    global _LOG_LEVEL
    _LOG_LEVEL = logLevel

    tf.logging.set_verbosity(tf.logging.ERROR)

    retError = [-1, -1]
    if not checkIfPathExists('log'):
        createFolder('log')
    tempFolderName = createFolder('log/log')

    # Read the input image 
    im = cv2.imread(imagePath)
    if im is None:
        return retError, None, None

    im_copy = cv2.imread(imagePath)    
    height, width, channels = im.shape

    if _LOG_LEVEL >= 4:
        print ("Image dimensions (hXw): " + str(height) + "X" + str(width))
        _printAxis(x_axis, y_axis, height, width)
    
    # Search digits in the given image
    pointList, recList = _searchDigitsInImage(im, im_copy, point_list, modelName, float(width * height), tempFolderName, x_axis, y_axis)
    # Collect numbers in groups
    numbersList = _collectDigitsInGroups(pointList, height, width)
    if len(numbersList) != 2:
        index = 0
        while index < len(numbersList):
            print (str(numbersList[index]))
            index = index + 1
        return retError, None, None
    l = _findXAndYValues(numbersList)

    if logLevel == 0:
        deleteFolder(tempFolderName)
        tempFolderName = None, None
    elif logLevel == 1:
        deleteAllFilesByTypeInFolder(tempFolderName, "png")
    return l, recList, tempFolderName
    
### Run as main for DEBUG ###
if __name__ == "__main__":
    import sys
    l, recList, folderName = recognize(sys.argv[1])
    print (str(l))
    
