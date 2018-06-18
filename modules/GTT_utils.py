import os
import math
from PIL import Image
import shutil

'''
Check if the given path represents a file in the system
'''
def checkIfFileExists(filePath):
    if not os.path.isfile(filePath):
        return False
    return True

'''
calculate Euclidean distance in R2
'''
def calcDistance(x1, y1, x2, y2):
    return math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))

'''
calculate Euclidean distance in R2.
pt1/pt2 - (x,y)
'''
def calcDistancePoints(pt1, pt2):
    return calcDistance(pt1[0], pt1[1], pt2[0], pt2[1])

'''
the function create a new folder in the current directory.
If the folder name in use, the function will add number as a suffix.
folderName default value is ".temp".
'''
def createFolder(folderName = ''):
    if folderName == '':
        folderName = '.temp'
    index = ""
    while os.path.exists(folderName + str(index)):
        if index == "":
            index = 0
        else:
            index = index + 1
    name = folderName + str(index)
    os.makedirs(name)
    return name

'''
The function delete folder
'''
def deleteFolder(folderPath):
    if os.path.exists(folderPath):
        shutil.rmtree(folderPath, ignore_errors=True)

'''
The function retunr a valid imgName that doesn't exists in the system
'''
def getEmptyImageName(folderPath, imgName = 'a', extension = 'png', startWithNumer = False):
    index = ""
    if startWithNumer:
        index = 0
    while os.path.exists(folderPath + "/" + imgName + str(index) + "." + extension):
        if index == "":
            index = 0
        else:
            index = index + 1
    return folderPath + "/" + imgName + str(index) + "." + extension

'''
if the original image is in JPEG format - converts it to PNG.
'''
def convertImageToPNG(source):
    create_image = False
    
    if not checkIfFileExists(source):
        return None, None

    filename, file_extension = os.path.splitext(str(source))
    if file_extension.lower() != ".png":
        im = Image.open(source)
        im.save(filename + '.png')
        source = filename + '.png'
        create_image = True
    return source, create_image

'''
the function append text to given file. If the file doesn't exists in the system,
it will creates it.
'''
def appendToFile(filePath, text):
    with open(filePath, "a+") as myfile:
        myfile.write(text)

'''
the function delete a file
'''
def deleteFile(f):
    if checkIfFileExists(f):
        os.remove(f)

'''
The function delete all files in the given folder with a given extension.
In case if extension = None, the function will delete all files in the directory
'''
def deleteAllFilesByTypeInFolder(folderPath, extension):
    folderItems = os.listdir(folderPath)
    for item in folderItems:
        if extension is None or item.endswith(extension):
            deleteFile(os.path.join(folderPath, item))

'''
the function return file name and extension of a given file
'''
def getFileNameAndExtension(img):
    filename, file_extension = os.path.splitext(str(img))
    filename = filename.replace('/','\\')
    filename = filename.split('\\')
    return filename, file_extension

'''
The function check if the path exists (file/directory)
'''
def checkIfPathExists(path):
    return os.path.exists(path)