import os
import sys
import random
import math
import re
import time
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
MRCNN_model_path = "prediction_model\\mask_rcnn_custom_0005.h5"
model = modellib.MaskRCNN(mode="inference", model_dir=MRCNN_model_path, config=Config())
#load model weights
model.load_weights(MRCNN_model_path, by_name=True)

save_directory = './out'
input_directory = './input'

img_name_list = []
avg_viability_list =[]
avg_circularity_list = []

for filename in os.listdir(input_directory):
    print("image_name: ",filename)
    f = os.path.join(input_directory, filename)
    if os.path.isfile(f):
       # print(f)
        image = cv2.imread(f)
        results1 = model.detect([image], verbose=1)
        r1 = results1[0]
        thrld = 0.1
        dr = save_directory + '/' + filename
        """ Analysis part """
        cell_count,cell_state_list,circularity_list = visualize_custom.display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],
                                r1['scores'], thrld, dr)
        
        avg_live_state = np.round(mean(cell_state_list))
        avg_circularity = mean(circularity_list)
        avg_circularity = round(avg_circularity, 3)
        
        #print(r1['masks'])
        print("cellcount: ",cell_count)
        print("cell_liveliness_state: ",cell_state_list)
        print("circularity;",circularity_list)
        print("average_viability: ",avg_live_state)
        print("average_circularity: ",avg_circularity)
        
        img_name_list.append(filename)
        avg_viability_list.append(avg_live_state)
        avg_circularity_list.append(avg_circularity)

        """ csv file creation """
        csv_lst = [img_name_list,avg_viability_list,avg_circularity_list]
        
        df = pd.DataFrame(list(zip(img_name_list, avg_circularity_list,avg_viability_list)),
               columns =['Image_Name', 'Average_Circularity(ideal circle=0)','Average_Viability (%)'])
        df.to_csv("Analysis_out.csv", encoding='utf-8',index =False)
        
        

       
