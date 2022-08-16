import argparse
import enum
import os
import json
import random
import shutil
parser = argparse.ArgumentParser(
        description='Mask R-CNN infrence for organoid counting and segmentation')
parser.add_argument('--input', help='Path to input directory')
parser.add_argument('--out', help='Path to output directory')
parser.add_argument('--count', help='how many random images to pull')
args = parser.parse_args()
assert args.input is not None, "Please provide an input directory"
assert args.out is not None, "Please provide an output directory"
input_directory = args.input
save_directory = args.out
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
input("!!!Warning!!! this action will CLEAR the output directory. Press any key to continue or Ctrl-C to cancel")
# delete all files in output dir
for file in os.listdir(save_directory):
    file_path = os.path.join(save_directory, file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)
# list all files in input directory
img_name_list = os.listdir(input_directory)
randomImages = random.sample(img_name_list, int(args.count))
mappings = {}
for index, filename in enumerate(randomImages):
    img_path = os.path.join(input_directory, filename)
    save_path = os.path.join(save_directory, f"image_{index}.tiff")
    shutil.copy(img_path, save_path)
    mappings[filename] = f"image_{index}.tiff"
# make a csv from mappings
with open(os.path.join(save_directory, "mappings.csv"), "w") as f:
    for key, value in mappings.items():
        f.write(f"{key},{value}\n")
with open(os.path.join(save_directory, "mappings.json"), "w") as f:
    f.write(json.dumps(mappings))