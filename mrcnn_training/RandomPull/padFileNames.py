import argparse
from genericpath import isfile
import os

parser = argparse.ArgumentParser(
        description="Pad file names with 0's")
parser.add_argument('--input', help='Path to input directory')
args = parser.parse_args()
assert args.input is not None, "Please provide an input directory"
inputDirectory = args.input

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
