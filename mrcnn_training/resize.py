import cv2
import os
import argparse

parser = argparse.ArgumentParser(
        description='Image resizing') 
parser.add_argument('--input', help='Path to input directory')
parser.add_argument('--out', help='Path to output directory')
args = parser.parse_args()
assert args.input is not None, "Please provide an input directory"
assert args.out is not None, "Please provide an output directory"
path = args.input
save_directory = args.out


os.makedirs(f"{path}", exist_ok=True)
for file in os.listdir(path):
    f = os.path.join(path, file) 
    print(f"Resizing {f}")
    img = cv2.imread(f)
    # resized = cv2.resize(img,(1360,1024), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(img,None,fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    file = file.split(".")[0] + ".png"
    newName = f"{save_directory}/{file}"
    print(newName)
    cv2.imwrite(newName, resized)
    



