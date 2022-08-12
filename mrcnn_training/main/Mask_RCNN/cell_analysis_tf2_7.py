import os
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


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



#from mrcnn.visualize_custom import display_images

class Config(Config):
    NAME = "model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 2
    DETECTION_MIN_CONFIDENCE = 0.1
    # IMAGE_MIN_DIM = 832
    # IMAGE_MAX_DIM = 832

config = Config()
print(config)
config.display()
#print(kkk)
MRCNN_model_path = "prediction_model\\Workingmask_rcnn_custom_0015.h5"
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

save_directory = './out'
input_directory = './input'
img_name_list = []
avg_viability_list =[]
avg_circularity_list = []
avg_contour_area_list = []
image_resolution_list = []

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
        cell_count,cell_state_list,circularity_list, contour_area_list = visualize_custom.display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],
                                r1['scores'], thrld, dr,Debug=False)

        print(len(cell_state_list))
        print(contour_area_list)
        print(len(contour_area_list))
        avg_live_state =np.average(cell_state_list,weights=contour_area_list)
        avg_circularity = mean(circularity_list)
        avg_circularity = round(avg_circularity, 3)
        avg_contour_area = mean(contour_area_list)*0.20112673401606182 #convert to um
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

        """ csv file creation """
        csv_lst = [img_name_list,image_resolution_list,avg_viability_list,avg_contour_area_list,avg_circularity_list]

        df = pd.DataFrame(list(zip(img_name_list,image_resolution_list,avg_circularity_list,avg_contour_area_list,avg_viability_list)),
               columns =['Image_Name','Image_Resolution','Average_Circularity(ideal circle=0)','Average area','Average_Viability (%)'])

        print(df)
        df.to_csv("Analysis_output.csv", encoding='utf-8',index =False)