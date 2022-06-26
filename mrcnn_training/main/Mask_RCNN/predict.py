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
from mrcnn import visualize_predict
from mrcnn import utils
from mrcnn.visualize_predict import display_images


class Config(Config):
    NAME = "model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 2
    DETECTION_MIN_CONFIDENCE = 0.1
    # IMAGE_MIN_DIM = 832
    # IMAGE_MAX_DIM = 832


config = Config()
MRCNN_model_path = "prediction_model\\mask_rcnn_custom_0001.h5"
model = modellib.MaskRCNN(mode="inference", model_dir=MRCNN_model_path, config=Config())
# load model weights
model.load_weights(MRCNN_model_path, by_name=True)

save_directory = './out'
input_directory = './input'

for filename in os.listdir(input_directory):
    f = os.path.join(input_directory, filename)
    if os.path.isfile(f):
        print(f)
        image = cv2.imread(f)
        results1 = model.detect([image], verbose=10)

        r1 = results1[0]
        thrld = 0.1
        dr = save_directory + '/' + filename
        cell_count = visualize_predict.display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],
                                                         r1['scores'], thrld, dr)
        # print(r1['masks'])
        print("count", cell_count)

        """
        for i in range(r1['masks'].shape[-1]):
            mask = r1['masks'][:, :, i]
            image[mask] = 255
            image[~mask] = 0
            unique, counts = np.unique(image, return_counts=True)
            print("unique",unique,counts)
            mask_area = counts[1] / (counts[0] + counts[1])
            #print(counts[1])
            positive_pixel_count = mask.sum() # assumes binary mask (True == 1)
            h, w = mask.shape # assumes NHWC data format, adapt as needed
            area = positive_pixel_count / (w*h)
            print("area",area)
        #print(kkk)

        # Get predictions of mask head
        mrcnn = model.run_graph([image], [
            ("detections", model.keras_model.get_layer("mrcnn_detection").output),
            ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ])

        # Get detection class IDs. Trim zero padding.
        det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
        det_count = np.where(det_class_ids == 0)[0][0]
        det_class_ids = det_class_ids[:det_count]

        #print("{} detections: {}".format(
            #det_count, np.array(dataset.class_names)[det_class_ids]))
        # Masks
        det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
        det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c] 
                                      for i, c in enumerate(det_class_ids)])
        det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                              for i, m in enumerate(det_mask_specific)])
        #log("det_mask_specific", det_mask_specific)
        #log("det_masks", det_masks)
        #display_images(det_mask_specific[:15] * 255, cmap="gray", interpolation="none")
        """
