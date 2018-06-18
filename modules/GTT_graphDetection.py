import cv2
import numpy as np

### Functions ###

'''
Helper class. This class represents an abstract and easy modified graph line -
can add points and get all points.
'''
class _GraphLine():
    def __init__(self):
        self._points = []

    def addPoint(self,x,y):
        self._points.append((x,y))

    def getPoints(self):
        return list(self._points)

'''
Visual scan of the picture - preforms Hough Line Transform and returns it together
with the physical dimensions of the image.
'''
def _visualPicRead(source):
    img = cv2.imread(source)
    height, width, channels = img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)

    min_line_length = 5  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    return cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap), height, width

'''
Helper func - verifies the given dot is valid for output.
'''
def _verifyDot(pt,rec_list,x_axis=None,y_axis=None):
    EPSILON = 15

    # No points on axes
    if not (x_axis is None or y_axis is None):
        sorted_x_axis = sorted(x_axis,key=lambda pt:pt[1])
        sorted_y_axis = sorted(y_axis,key=lambda pt:pt[0])
        
        x_axis_y_min = sorted_x_axis[0][1]
        x_axis_y_max = sorted_x_axis[-1][1]
        y_axis_x_min = sorted_y_axis[0][0]
        y_axis_x_max = sorted_y_axis[-1][0]
        
        if (pt[1] >= (x_axis_y_min-EPSILON) and pt[1] <= (x_axis_y_max+EPSILON)) or (pt[0] >= (y_axis_x_min-EPSILON) and pt[0] <= (y_axis_x_max+EPSILON)):
                return False

    # Filters out all points inside rectangles.
    if not rec_list is None:  
        for rec in rec_list:
            if pt[0] >= (rec[0][0]-EPSILON) and pt[0] <= (rec[1][0]+EPSILON) and pt[1] >= (rec[0][1]-EPSILON) and pt[1] <= (rec[1][1]+EPSILON):
                return False
    return True

'''
Returns all relevant graph line points for the output in the following format:
[All relevant points, X maximum - pixel values, Y maximum - pixel values]
'''
def getGraphPoints(source,x_axis,y_axis,rec_list):
    points,_,_ = getGraphRaw(source,rec_list)

    sorted_points = sorted(points, key=lambda y: y[1])
    filtered_points = [pt for pt in sorted_points if _verifyDot(pt,rec_list,x_axis,y_axis)]

    if len(filtered_points) == 0:
        filtered_points = None
        x_pixel_max = None
        y_pixel_max = None
    else: 
        x_pixel_max = np.array([filtered_points[0][0],0])
        y_pixel_max = np.array([0,filtered_points[0][1]])
    return filtered_points, x_pixel_max, y_pixel_max

'''
Strips the graph into basic points of interests - [dots,height,width]
'''
def getGraphRaw(source, rec_list = None):
    lines, height, width = _visualPicRead(source)

    curr_line = _GraphLine()
    for line in lines:
        for x1,y1,x2,y2 in line:
            if _verifyDot((x1,y1), rec_list):
                curr_line.addPoint(x1,y1)
            if _verifyDot((x2,y2), rec_list):
                curr_line.addPoint(x2,y2)

    points = []
    points = curr_line.getPoints()
    if len(points) == 0:
        points = None
    return points, height, width