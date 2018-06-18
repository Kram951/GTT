import numpy as np
import sys
import argparse

from modules.GTT_graphDetection import getGraphPoints,getGraphRaw
from modules.GTT_numbersDetection import recognize
from modules.GTT_stripes import getGraphAxesAndOrig
from modules.GTT_output import createEXCEL
from modules.GTT_utils import calcDistance
from modules.GTT_utils import convertImageToPNG
from modules.GTT_utils import createFolder
from modules.GTT_utils import deleteFile
from modules.GTT_utils import getFileNameAndExtension
from modules.GTT_utils import checkIfFileExists
from modules.GTT_utils import checkIfPathExists

'''
Final stop - exits the program.
'''
def stopRun(out = None):
    if out is None:
        print("Error occurred")        
    else:
        print(out)        
    exit(0)

''' 
Main func - 
receives the source-image and connects between visual recognition of graph line and numbers.
returns list of graph points scaled to the wanted x-y values.
'''
def pic2data(source,logLevel,x_y_vals = None):

    points,height,width = getGraphRaw(source)
    if points is None:
        stopRun("Line recognition failed")

    x_axis,y_axis,orig_place = getGraphAxesAndOrig(points,height,width)

    visibile_points, rec_list, _ = recognize(source,None, x_axis,y_axis,logLevel)
    if visibile_points == [-1,-1] and x_y_vals is None:
        stopRun("Visual recognition of numbers failed")

    if logLevel > 0:
        print("Visual points recognized: {0}".format(visibile_points))
        
    all_points = getGraphPoints(source,x_axis,y_axis,rec_list)
    
    if all_points[2] is None:
        stopRun("Line recognition failed")

    origin = orig_place
    origin_value = (0.0,0.0)
    
    Xref =  all_points[1]
    Xref_value = visibile_points[0]
    
    Yref =  all_points[2]
    Yref_value = visibile_points[1]

    if not x_y_vals is None:
        Xref_value = x_y_vals[0]
        Yref_value = x_y_vals[1]

    # Assume the graph axes are straight -     
    Xref[1] = origin[1]
    Yref[0] = origin[0]     
     
    selected_points = all_points[0]
    
    # Transform to the correct graph values
    OXref = Xref - origin
    OYref = Yref - origin
    xScale =  (Xref_value - origin_value[0]) / np.linalg.norm(OXref)
    yScale =  (Yref_value - origin_value[1]) / np.linalg.norm(OYref)
     
    ux = OXref / np.linalg.norm(OXref)
    uy = OYref / np.linalg.norm(OYref)
    
    result = [(ux.dot(pt - origin) * xScale + origin_value[0],
               uy.dot(pt - origin) * yScale + origin_value[1])
               for pt in selected_points]
    
    return result

'''
Receieves a dot list and formats it to the wanted out template.
Returns the formatted output.
'''
def exportXY(dot_list,x_scale):

    '''
    Return the y_val for the wanted_x - assuming that (x1,y1),(x2,y2)
    creates a straight line
    '''
    def calcY_ValOnStraightLine(wanted_x,x1,y1,x2,y2):
        # y = mx + n
        m = (y2-y1)/(x2-x1)
        n = y1-(m*x1)
        return (m*wanted_x)+n

    sorted_res = sorted(dot_list, key=lambda x: x[0])
    
    tmp_out = {}
    for xi, yi in sorted_res:
        tmp_out[xi] = yi 

    x = [xi for xi,_ in sorted_res]
    #y = [yi for _,yi in sorted_res]

    output = {}

    top_rng = int(max(x))+1
    for i in range(0,top_rng,x_scale):
        if i in x:
            output[i] = tmp_out[i]
        else:
            x_before = y_before = x_after = y_after = -1
            for val in sorted_res:
                if val[0] > i:
                    x_after = val[0]
                    y_after = val[1]
                    break
                x_before = val[0]
                y_before = val[1]
            if x_before == -1:
                output[x_after] = y_after
            elif x_after == -1:
                output[x_before] = y_before
            else:
                output[i] = calcY_ValOnStraightLine(i,x_before,y_before,x_after,y_after)
    
    return output

if __name__=="__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    parser=argparse.ArgumentParser(
        description='''GraphToTable: receives image as input and produces excel file with the output.''')
    parser.add_argument('--source', nargs=1, type=str, default=None, help='The path for the image source file')
    parser.add_argument(
        '--log_level', type=int, default=0,
        help='''0 - No log info ;
             1 - Only log file ;
             2 - Log files and images'''
        )
    parser.add_argument('--scale', type=int, default=1, help='Output scale for x axis.')
    parser.add_argument(
        '--x_y_max', nargs=2, type=int, default=None, help=
        'Value for (x_max,y_max) - use to avoid numbers recognition.'
        )
    args=parser.parse_args()
    
    logLevel = args.log_level

    if logLevel > 0:
        print("Args: " + str(args))

    source = args.source
        
    if source is None:
        stopRun("No source file received")

    source = source[0].replace("'","")
    if logLevel > 0:
        print ("Source " + source)
    if not checkIfFileExists(source):
        stopRun("Source file received doesn't exist")

    filename, file_extension = getFileNameAndExtension(source)
    
    create_image = False
    if file_extension.upper() != ".PNG":
        source, create_image = convertImageToPNG(source)
        if source is None:
            stopRun("Couldn't convert image to PNG")    

    print("Starting data extraction from the image")
    dot_list = pic2data(source,logLevel,args.x_y_max)

    print("Editing final dot list")
    modified_dot_list = exportXY(dot_list,args.scale)
    
    folder_name = 'out'
    
    if not checkIfPathExists(folder_name):
        createFolder(folder_name)
    
    print("Creating the excel file")

    index = 0
    f_name = folder_name + '\\' + filename[-1] + "_"
    while checkIfFileExists((f_name + str(index) + '.xlsx')):
        index = index + 1

    createEXCEL(f_name + str(index), modified_dot_list)

    if create_image:
        deleteFile(source)
    
    print("Done")