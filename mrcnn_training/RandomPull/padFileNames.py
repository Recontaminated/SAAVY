inputDirectory = r"C:\Users\minec\OneDrive\Documents\GitHub\kylieDataAnylasis\mrcnn_training\main\Mask_RCNN\input"
from genericpath import isfile
import os

from cv2 import split
for file in os.listdir(inputDirectory):
    # get the file name
    if not isfile(os.path.join(inputDirectory, file)):
        print("skipping file: ", file)
        continue
    fileName = os.path.splitext(file)[0][6:]
    # remove last character
    print("fileName: ", fileName)
    fileName = fileName[:-1]
    print("fileName: ", fileName)
    # add 0 to the front

    fileName = fileName.zfill(3)
    rename = "image_" + fileName + ".jpg"
    print("renaming: ", file, " to: ", rename)
    os.rename(os.path.join(inputDirectory, file), os.path.join(inputDirectory, rename))
