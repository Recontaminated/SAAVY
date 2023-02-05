from ast import parse
import os
import pstats
import sys
import random
import math
import re
import time
from turtle import width
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import csv
from statistics import mean
import pandas as pd

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize_custom
from mrcnn import utils
import cProfile

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import argparse
parser = argparse.ArgumentParser(
        description='Mask R-CNN infrence for organoid counting and segmentation')
parser.add_argument('--input', help='Path to input directory')
parser.add_argument('--out', help='Path to output directory')


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



#from mrcnn.visualize_custom import display_images

class Config(Config):
    NAME = "model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 2
    DETECTION_MIN_CONFIDENCE = 0.7
    # IMAGE_MIN_DIM = 832
    # IMAGE_MAX_DIM = 832

config = Config()
print(config)
config.display()
#print(kkk)
MRCNN_model_path = "prediction_model\\93Trained.h5"
"""
from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
"""


model = modellib.MaskRCNN(mode="inference", model_dir=MRCNN_model_path, config=Config())
#load model weights
model.load_weights(MRCNN_model_path, by_name=True)

"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

tf.keras.Model.load_weights(model.keras_model, MRCNN_model_path, by_name=True)
"""
args = parser.parse_args()
if args.input is None:
    input_directory = './input'
else:
    input_directory = args.input
if args.out is None:
    save_directory = './out'
else:
    save_directory = args.out

#setup all output lists for making df

img_name_list = []
avg_viability_list =[]
avg_circularity_list = []
avg_contour_area_list = []
organoidCoveredArea_list = []
image_resolution_list = []
avg_raw_viability_list = []
count_list = []
background_list = []
organoidSmallestList = []
organoidLargestList = []
lowest_viability_list = []
highest_viability_list = []
print(os.listdir(input_directory))
for filename in os.listdir(input_directory):
    print("image_name: ",filename)
    f = os.path.join(input_directory, filename)
    if os.path.isfile(f):
    # print(f)
        image = cv2.imread(f)
        width, height, channels = image.shape
        image_resolution_list.append(f"{width}x{height}")

        results1 = model.detect([image], verbose=10)
        r1 = results1[0]
        thrld = 0.1
        dr = save_directory + '/' + filename
        """ Analysis part """
        """ Analysis part """
        cellCount,cell_state_list,circularity_list, contour_area_list, bkgrnd, bkgrnd_area_analized = visualize_custom.display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],
                                r1['scores'], thrld, dr,Debug=False) or ([-1], [-1], [-1], [-1], [-1])
        if cellCount == 0:
            img_name_list.append(filename)
            avg_viability_list.append(-1)
            avg_circularity_list.append(-1)
            avg_contour_area_list.append(-1)
            avg_raw_viability_list.append(-1)
            background_list.append(-1)
            count_list.append(0)
            organoidCoveredArea_list.append(0)
            organoidSmallestList.append(-1)
            organoidLargestList.append(-1)
            lowest_viability_list.append(-1)
            highest_viability_list.append(-1)
            
        else:     
            avg_live_state =np.average(cell_state_list,weights=contour_area_list)
            # avg_live_state = np.average(cell_state_list)
            avg_circularity = mean(circularity_list)
            avg_circularity = round(avg_circularity, 3)
            avg_contour_area = mean(contour_area_list)*0.20112673401606182 #convert to um
      # TODO: ADD PLACES THAT WILL BREAK IF RESOLUTION CHANGEs
            organoidCoveredArea_list.append((sum(contour_area_list)/bkgrnd_area_analized)*100)
            avg_contour_area = round(avg_contour_area, 3)
            #print(r1['masks'])
            # print("cellcount: ",cell_count)
            # print("cell_liveliness_state: ",cell_state_list)
            # print("circularity;",circularity_list)
            # print("average_viability: ",avg_live_state)
            # print("average_circularity: ",avg_circularity)
            
            img_name_list.append(filename)
            avg_viability_list.append(avg_live_state)
            avg_circularity_list.append(avg_circularity)
            avg_contour_area_list.append(avg_contour_area)
            count_list.append(len(cell_state_list))
            avg_raw_viability_list.append(mean(cell_state_list))
            background_list.append(bkgrnd)
            organoidSmallestList.append(min(contour_area_list))
            organoidLargestList.append(max(contour_area_list))
            lowest_viability_list.append(min(cell_state_list))
            highest_viability_list.append(max(cell_state_list))
        """ csv file creation """
        print(organoidCoveredArea_list)
        df = pd.DataFrame(list(zip(img_name_list,image_resolution_list,count_list,organoidCoveredArea_list,organoidSmallestList,organoidLargestList,lowest_viability_list,highest_viability_list,avg_circularity_list,avg_contour_area_list,background_list, avg_raw_viability_list, avg_viability_list)),
            columns =['Image_Name','Image_Resolution',"Organoid Count","image pct Analyzed",'Smallest organoid (px)','largest organoid (px)','lowest viability (pct)','highest viability (pct)','Average_Circularity(ideal circle=0)','Average area um (calculated)',"Background raw intensity (0-255) ","Average raw intenisty (0-255)",'Average_Viability (pct)'])

 
        df.to_csv(save_directory + "Analysis_output2.csv", encoding='utf-8',index =False)
