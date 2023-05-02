import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import time
import typing
from tqdm import tqdm

def buildModel(numClasses):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    inFeatures = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(inFeatures, numClasses)
    inFeaturesMask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hiddenLayer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(inFeaturesMask, hiddenLayer, numClasses)
    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

print("Loading model...")
model = torch.load("mask-rcnn-pedestrian.pt")
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
        return [], [], []

    pred_t = filtered_pred_indices[
        -1]  # it is the index of the last prediction that has a score higher than the confidence threshold

    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class




def segment_instance(img_path: str, confidence=0.5, rect_th=2, text_size=2, text_th=2) -> np.ndarray:
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
    masks, boxes, pred_cls = get_prediction(img_path, confidence)

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
        cv2.drawContours(img, contours, -1, (0, 255, 0), rect_th)

        #get the average intensity of all pixels within mask
        imgSave = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgSave = imgSave * masks[i]
        viability, circularity, averageIntensity = analyzeCell(imgSave, backgroundIntesity)
        cv2.putText(img, str(round(viability, 2)), (int(x+ 10), int(y + 20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        masked_image_out = cv2.putText(img, str(round(averageIntensity, 2)), (int(x + 20), int(y + 40)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cells.append(viability)
    print(np.average(cells))

    return img


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
    masked = imgSave[imgSave != 0] # ignore the completely black background
    masked = masked[masked != 255]#ignore the status bar white
    avg = np.average(masked)

    return avg

def analyzeCell(cell, backgroundIntensity):
    """

    :param cell: organoid
    :return: average viability, area, circulatriy
    """
    cell = cell[cell != 0] # ignore the completely black background
    averageIntensity = np.average(cell)
    cell_state = ((60 - np.clip((backgroundIntensity - 15 - averageIntensity), 0, 60)) / 60) * 100

    circularity = cv2.Laplacian(cell, cv2.CV_64F).var()


    return cell_state, circularity, averageIntensity





folder = r"C:\Users\minec\Downloads\20230405_Longevity_Exports\20230405_Longevity_Exports"
# folder = r"validationData"
files = os.listdir(folder)

timeStart = time.time()
# for file in files:
images = []
for file in tqdm(files):
    if not (file.endswith('.jpg') or file.endswith('.png')):
        continue
    print(file)
    segment_instance(folder + "\\" + file, confidence=0.8)


timeEnd = time.time()
print("time taken: ", timeEnd - timeStart)
print("time taken per image: ", (timeEnd - timeStart) / len(files))

# plt show all images
for img in images:
    print("showing")
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

