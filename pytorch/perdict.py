from typing import Tuple, List, Dict, Union, Any, Optional, Iterable

import torch

import os
import time
import typing
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help="folder of images to analyze")
parser.add_argument("output", help="folder to save output images")
args = parser.parse_args()
if args.input is None or args.output is None:
    print("please provide input and output folders")
    exit()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')

print("Loading model...")
model = torch.load("torchDebug01.1.pt")
# model = torch.load("debug01.pt")
print("Model loaded.")
print("running on device: ", device)
model.to(device)


model.eval()
CLASS_NAMES = ["__background__", "cell"]
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
import random
import warnings

warnings.filterwarnings("ignore")


def get_prediction(img_path, confidence):
    img = Image.open(img_path).convert("RGB")  # get rid of alpha channel
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]["scores"].detach().cpu().numpy())
    filtered_pred_indices = [pred_score.index(x) for x in pred_score if x > confidence]

    if not filtered_pred_indices:
        return [], [], [], None

    pred_t = filtered_pred_indices[
        -1
    ]  # it is the index of the last prediction that has a score higher than the confidence threshold

    masks = (pred[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()
    # print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]["labels"].cpu().numpy())]
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]["boxes"].detach().cpu().numpy())
    ]
    masks = masks[: pred_t + 1]
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]
    confidence_scores = pred_score[: pred_t + 1]
    return masks, pred_boxes, pred_class, confidence_scores


def segment_instance(
    img_path: str, confidence_thresh=0.5, rect_th=2, text_size=2, text_th=2
) -> tuple[Any, Any]:
    masks, boxes, pred_cls, confidence_scores = get_prediction(
        img_path, confidence_thresh
    )

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cells = []
    backgroundIntesity = calcBackgroundIntensity(img, masks)

    for i in range(len(masks)):
        pt1 = tuple(map(int, boxes[i][0]))
        pt2 = tuple(map(int, boxes[i][1]))

        x, y = pt2

        contours, _ = cv2.findContours(
            masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            continue
        (x, y), (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(
            max(contours, key=cv2.contourArea)
        )

        semi_major_axis = majorAxisLength / 2
        semi_minor_axis = minorAxisLength / 2
        circularity = round(
            np.sqrt(pow(semi_major_axis, 2) - pow(semi_minor_axis, 2))
            / semi_major_axis,
            2,
        )
        # Draw the contours on the image
        cv2.drawContours(img, contours, -1, (0, 255, 0), rect_th)

        # get the average intensity of all pixels within mask
        imgSave = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgSave = imgSave * masks[i]

        viability, averageIntensity, area = analyzeCell(imgSave, backgroundIntesity)
        if area == 0:
            continue
        cv2.putText(
            img,
            str(round(viability, 2)),
            (int(x + 10), int(y + 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            img,
            str(confidence_scores[i]),
            (int(x + 10), int(y + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        masked_image_out = cv2.putText(
            img,
            str(round(averageIntensity, 2)),
            (int(x + 20), int(y + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        cell_meta = {
            "viability": viability,
            "circularity": circularity,
            "averageIntensity": averageIntensity,
            "area": area,
        }
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
    # flatten 2d array to 1d
    # plt.imshow(imgSave, cmap="gray")
    # plt.show()
    imgSave = imgSave.flatten()

    masked = imgSave[imgSave > 5]  # ignore the completely black background

    masked = masked[masked != 255]  # ignore the status bar white
    # ignore everything greater than 250

    avg = np.average(masked)
    print(avg)

    return avg


def analyzeCell(cell, backgroundIntensity):
    area = np.count_nonzero(cell)
    cell = cell[cell != 0]  # ignore the completely black background
    averageIntensity = np.average(cell)
    cell_state = (
        (60 - np.clip((backgroundIntensity - 15 - averageIntensity), 0, 60)) / 60
    ) * 100

    # circularity = 0

    return cell_state, averageIntensity, area


folder = args.input


files = os.listdir(folder)

timeStart = time.time()
# for file in files:
images = []
# make a new pandas DF
images_meta = []

for file in tqdm(files):
    if not (file.endswith(".jpg") or file.endswith(".png")):
        continue
    print(file)
    image, cells, backgroundIntensity = segment_instance(
        folder + "\\" + file, confidence_thresh=0.8
    )
    image_total_px = image.shape[0] * image.shape[1]
    sum_area = sum([cell["area"] for cell in cells])
    pct_area = sum_area / image_total_px

    images_meta.append(
        {
            "file": file,
            "cells": cells,
            "backgroundIntensity": backgroundIntensity,
            "pct_area_analyzed": pct_area,
        }
    )
    images.append(image)


timeEnd = time.time()
print("time taken: ", timeEnd - timeStart)
print("time taken per image: ", (timeEnd - timeStart) / len(files))

# path = os.path.join(folder, "out")
path = args.output
if not os.path.exists(path):
    os.mkdir(path)
for i in range(len(images)):
    try:
        cv2.imwrite(os.path.join(path, files[i]), images[i])
    except:
        print("error saving image: ", files[i])

# make a new dataframe with empty everything
df = pd.DataFrame(
    columns=[
        "file",
        "count",
        "avg_viability",
        "avg_circularity",
        "avg_intensity",
        "radius (area / pi)",
    ]
)
for image in images_meta:
    if image["cells"] == []:
        avg_viability = -1
        avg_circularity = -1
        avg_intensity = -1
        avg_area = -1

    else:
        num_cells = len(image["cells"])
        avg_viability = np.average(
            [cell["viability"] for cell in image["cells"]]
        ).round(2)
        avg_circularity = np.average(
            [cell["circularity"] for cell in image["cells"]]
        ).round(5)
        avg_intensity = np.average(
            [cell["averageIntensity"] for cell in image["cells"]]
        ).round(2)
        avg_area = np.average([cell["area"] for cell in image["cells"]]).round(2)

    df = df.append(
        {
            "file": image["file"],
            "count": num_cells,
            "pct_analyzed": image["pct_area_analyzed"] * 100,
            "background_intenstiy": image["backgroundIntensity"],
            "avg_viability": avg_viability,
            "avg_circularity": avg_circularity,
            "avg_intensity": avg_intensity,
            "avg_area": avg_area,
        },
        ignore_index=True,
    )
try:
    df.to_csv(os.path.join(path, "summary.csv"), index=False)
except PermissionError:
    print("Please close the summary.csv file. press any key to continue")
    input()
    # df.to_csv("out\\summary.csv", index=False)
