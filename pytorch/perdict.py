from typing import Tuple, List, Dict, Union, Any, Optional, Iterable

import torch

import os
import time
import typing
from tqdm import tqdm
import pandas as pd



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

print("Loading model...")
model = torch.load(r"C:\Users\minec\OneDrive\Documents\GitHub\kylieDataAnylasis\pytorch\models\trainedLongerModelv2.pt")
print("Model loaded.")
print("running on device: ", device)
model.to(device)


model.eval()
CLASS_NAMES = ['__background__', 'cell']
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
import random
import warnings

warnings.filterwarnings('ignore')


def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)

    return coloured_mask


def get_prediction(img_path, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0
    
    """
    img = Image.open(img_path).convert('RGB')  # get rid of alpha channel
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    filtered_pred_indices = [pred_score.index(x) for x in pred_score if x > confidence]

    if not filtered_pred_indices:
        return [], [], [], None

    pred_t = filtered_pred_indices[
        -1]  # it is the index of the last prediction that has a score higher than the confidence threshold

    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    confidence_scores = pred_score[:pred_t + 1]
    return masks, pred_boxes, pred_class, confidence_scores




def segment_instance(img_path: str, confidence_thresh=0.5, rect_th=2, text_size=2, text_th=2) -> tuple[
    Any, Any]:
    """
    segment_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    masks, boxes, pred_cls, confidence_scores = get_prediction(img_path, confidence_thresh)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cells = []
    backgroundIntesity = calcBackgroundIntensity(img, masks)


    for i in range(len(masks)):
        # rgb_mask = get_coloured_mask(masks[i])
        # img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        pt1 = tuple(map(int, boxes[i][0]))
        pt2 = tuple(map(int, boxes[i][1]))

        x, y = pt2
        #draw a polygon of the mask
        contours, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the image

        #get the average intensity of all pixels within mask
        imgSave = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgSave = imgSave * masks[i]
        viability, circularity, averageIntensity, area, raw_intensity = analyzeCell(imgSave, backgroundIntesity)

        if area == 0:
            print("file is nan: ", img_path)
            continue
        cv2.putText(img, str(round(viability, 2)), (int(x+ 10), int(y + 0)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, str(confidence_scores[i]), (int(x + 10), int(y + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        masked_image_out = cv2.putText(img, str(round(averageIntensity, 2)), (int(x + 20), int(y + 30)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.drawContours(img, contours, -1, (0, 255, 0), rect_th)
        if viability == np.nan:
            print("file is nan: ", img_path)
        cell_meta = {"viability": viability, "circularity": circularity, "averageIntensity": averageIntensity, "area": area, "raw_intensity": raw_intensity}
        cells.append(cell_meta)
    print("background intensity: ", backgroundIntesity)
    
    return img, cells, backgroundIntesity


def calcBackgroundIntensity(img, masks) -> float:
    """
    :param img: image
    :param mask: a list of masks
    :return: average intensity of background
    """
    imgSave = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    combined_mask = np.zeros_like(imgSave)

    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)

    backgroundMask = np.logical_not(combined_mask)


    

    imgSave = imgSave * backgroundMask 
    # plt.imshow(imgSave)
    # plt.show()
    #flatten 2d array to 1d
    # plt.imshow(imgSave, cmap="gray")
    # plt.show()
    imgSave = imgSave.flatten()

    masked = imgSave[imgSave > 12] # ignore the completely black background
  
    masked = masked[masked != 255]#ignore the status bar white
    # ignore everything greater than 250
    
    avg = np.average(masked)
    print(avg)

    return avg

def analyzeCell(cell, backgroundIntensity):
    """

    :param cell: organoid
    :return: average viability, area, circulatriy
    """
    area = np.count_nonzero(cell)
    plt.imshow(cell, cmap="gray")
    cell = cell[cell > 0] # ignore the completely black background
    averageIntensity = np.average(cell)
    if backgroundIntensity < 180: #  backgtround has lots of noise
        cell_state = ((60 - np.clip((backgroundIntensity - 15 - averageIntensity), 0, 60)) / 60) * 100
    else:
        cell_state = ((50 - np.clip((backgroundIntensity - 25 - averageIntensity), 0, 50)) / 50) * 100

    # circularity = cv2.Laplacian(cell, cv2.CV_64F).var()
    circularity = 0

    if area == 0 or cell_state == np.nan or cell_state is None: #somehow model can output a cell with no area
        print("cell state is nan")
        return np.nan, np.nan, np.nan,0, np.nan
    

    return cell_state, circularity, averageIntensity, area, averageIntensity




if __name__ == '__main__':
    # folder = r"C:\Users\minec\Desktop\figure images"
    # folder = r"C:\Users\minec\Desktop\mappingsCSV"
    # folder = r"validationData"
    folder = r"C:\Users\minec\Downloads\20230405_Longevity_Exports\20230405_Longevity_Exports"
    files = os.listdir(folder)

    timeStart = time.time()
    # for file in files:
    images = []
    #make a new pandas DF
    images_meta = []

    for file in tqdm(files):
        if not (file.endswith('.jpg') or file.endswith('.png')):
            continue
        print(file)
        image, cells, backgroundIntensity = segment_instance(folder + "\\" + file, confidence_thresh=0.80)
        images_meta.append({"file": file, "cells": cells, "backgroundIntensity": backgroundIntensity})
        images.append(image)


    timeEnd = time.time()
    print("time taken: ", timeEnd - timeStart)
    print("time taken per image: ", (timeEnd - timeStart) / len(files))

    path = os.path.join(folder, "out")
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(len(images)):
        try:
            cv2.imwrite(os.path.join(path,files[i]), images[i])
        except:
            print("error saving image: ", files[i])

    #make a new dataframe with empty everything
    df = pd.DataFrame(columns=["file", "count", "avg_viability", "avg_circularity", "avg_intensity", "radius (area / pi)"])
    for image in images_meta:
        num_cells = len(image["cells"])

        if image["cells"] == []:
            avg_viability = -1
            avg_circularity = -1
            avg_intensity = -1
            avg_area = -1
            avg_radius = -1
            avg_raw_intensity = -1
        else:
            # print(image["cells"])
            # print(image["file"])
            avg_viability = np.average([cell["viability"] for cell in image["cells"]], weights=[cell["area"] for cell in image["cells"]]).round(2)
            # avg_viability = np.average([cell["viability"] for cell in image["cells"]]).round(2)
            avg_circularity = np.average([cell["circularity"] for cell in image["cells"]]).round(2)
            avg_intensity = np.average([cell["averageIntensity"] for cell in image["cells"]]).round(2)
            avg_area = np.average([cell["area"] for cell in image["cells"]]).round(2)
            avg_radius = (np.sqrt(avg_area / np.pi)).round(2)
            avg_raw_intensity = np.average([cell["raw_intensity"] for cell in image["cells"]]).round(2)


        df = df.append({"file": image["file"],"count": num_cells, "background_intenstiy" : image["backgroundIntensity"], "avg_viability": avg_viability, "avg_circularity": avg_circularity, "avg_intensity": avg_intensity, "radius (area / pi)": avg_radius, "raw_intensity":avg_raw_intensity}, ignore_index=True)
    try:
        df.to_csv(os.path.join(path,"summary.csv"), index=False)
    except PermissionError:
        print("Please close the summary.csv file. press any key to continue")
        input()
        # df.to_csv("out\\summary.csv", index=False)



