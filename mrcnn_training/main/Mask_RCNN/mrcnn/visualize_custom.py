from turtle import back, bk
import cv2
import os
import sys
import numpy as np
from skimage.measure import find_contours
import csv
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import Mask RCNN
from mrcnn import utils


############################################################
#  Visualization
############################################################

def apply_mask(image, mask, color, alpha=0):
    # Apply the given mask to the image.

    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def maskBackground(image, boxes, masks, class_ids, scores, threshold):
    # def display_instances(image, boxes, masks, class_ids, filename, threshold, scores=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    filename:image name
    scores: (optional) confidence scores for each box
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        return image

    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig_img = image.astype(np.uint8)
    masked_image = image.astype(np.uint32).copy()
    masked_image_fill = image.astype(np.uint32).copy()

    # dead = 0
    # live = 0
    cell = 0
    colors = [[0, 1, 0]]

    cell_state_list = []
    circularity_list = []
    for i in range(N):

        if scores[i] > threshold:
            color = colors[class_ids[i] - 1]
            if class_ids[i] - 1 == 0:
                cell = cell + 1
            # elif class_ids[i] - 1 == 1:
            # dead = dead + 1

            # Apply mask for each detected object
            mask = masks[:, :, i]
            # print("################################")
            # mask1 = 1*mask
            # print(mask1)S
            # cv2.imwrite("mask.jpg",mask1)

            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            # print(contours)
            masked_image = masked_image.astype(np.uint8)
            masked_image_fill = apply_mask(masked_image_fill, mask, color)
            masked_image_fill = masked_image_fill.astype(np.uint8)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                verts = np.array([verts], np.int32)
                verts = verts.reshape((-1, 1, 2))

                masked_image_fill = cv2.fillPoly(masked_image_fill, [verts], (255, 20, 20))
    return masked_image_fill


def get_bkgrnd_intensity(img_name, mask):
    # getting background intensity other than the mask region
    lower_range = np.array([110, 50, 50])
    upper_range = np.array([130, 255, 255])
    masked_image_fill = mask
    masked_image_fill_orig = mask
    # masked_image_fill_orig = cv2.resize(masked_image_fill_orig, (600, 600), interpolation=cv2.INTER_AREA)
    # masked_image_fill = cv2.resize(masked_image_fill, (600, 600), interpolation=cv2.INTER_AREA)
    masked_image_fill = cv2.cvtColor(masked_image_fill, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(masked_image_fill, lower_range, upper_range)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    mask1 = np.array(mask)
    coords = np.column_stack(np.where(mask1 == 0))
    # print(coords)
    bkgrnd_pix_list = []
    for i in coords:
        # print(i)
        x = i[0]
        y = i[1]
        # print(x,y)
        # image = cv2.circle(masked_image_fill_orig, (x, y), radius=0, color=(0, 0, 255), thickness=-1)

        pix_val = masked_image_fill_orig[x, y]
        pix_val = np.mean(pix_val)
        bkgrnd_pix_list.append(pix_val)
        # print("pixval...........",pix_val)
    # print(bkgrnd_pix_list)
    mask = np.abs(bkgrnd_pix_list) > 5
    # mask bkgrnd_pix_list using mask
    bkgrnd_pix_list = np.array(bkgrnd_pix_list)
    bkgrnd_pix_list = bkgrnd_pix_list[mask]

    bkgrnd_pix_avg_intensity = np.mean(bkgrnd_pix_list)
    # print(bkgrnd_pix_avg_intensity)
    # cv2.imshow('image', image)
    # cv2.waitKey()
    bkgrnd_pix_analyzed = len(bkgrnd_pix_list)
    return bkgrnd_pix_avg_intensity, bkgrnd_pix_analyzed


def display_instances(image, boxes, masks, class_ids, scores, threshold, filename, Debug=False):
    # def display_instances(image, boxes, masks, class_ids, filename, threshold, scores=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    filename:image name
    scores: (optional) confidence scores for each box  
    """
    # Number of instances  

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
        return 0, [], [], [], -1, -1
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    maskedBackground = maskBackground(image, boxes, masks, class_ids, scores, threshold)
    bkgrn, bkgrnd_pix_analyzed = get_bkgrnd_intensity(image,maskedBackground)

    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_image = image.astype(np.uint32).copy()
    # dead = 0
    # live = 0
    cell = 0
    colors = [[0, 1, 0]]

    cell_state_list = []
    circularity_list = []
    contour_area_list = []
    for i in range(N):

        if scores[i] > threshold:
            color = colors[class_ids[i] - 1]
            if class_ids[i] - 1 == 0:
                cell = cell + 1
            # elif class_ids[i] - 1 == 1:
            # dead = dead + 1

            # Apply mask for each detected object
            mask = masks[:, :, i]
            # print("################################")
            # mask1 = 1*mask
            # print(mask1)S
            # cv2.imwrite("mask.jpg",mask1)

            masked_image = apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)

            padded_mask[1:-1, 1:-1] = mask

            contours = find_contours(padded_mask, 0.5)
            # print(contours)
            masked_image = masked_image.astype(np.uint8)
            thickness = 4

            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)           
                verts = np.fliplr(verts) - 1
                verts = np.array([verts], np.int32)
                verts = verts.reshape((-1, 1, 2))
                clr = [0, 250, 0]
                masked_image_out = cv2.polylines(masked_image, [verts], True, clr, thickness)
                mask = np.zeros(grey_img.shape, np.uint8)

                cv2.drawContours(mask, [verts], 0, 255, -1)
                x, y, w, h = cv2.boundingRect(verts)
                masked_image_out = cv2.putText(masked_image_out, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                               (0, 0, 0), 2)
                # crop_img = orig_img[y:y + h, x:x + w]
                # cv2.imshow('cutted contour',crop_img)
                # cv2.waitKey()

                ################################## cell region properties calculation start ##########
                # reference: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html

                # print("avg_pixel_cell crop bb:",avg_pixel_cell)

                contour_area = cv2.contourArea(verts)
                contour_area_list.append(contour_area)
                # print("contour area: ",contour_area)

                # print("rect_area:",rect_area)

                # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grey_img,mask = mask)
                # #print("min_max values:",min_val,max_val)
                """ Average intensity value of a cell """
                # plt.imshow(mask, cmap='gray')
                # plt.show()
                mean_val = cv2.mean(grey_img, mask=mask)
                mean_val = list(mean_val)
                mean_val = round(mean_val[0])

                """eccentricity/circularity of a circle is 0"""
                (x, y), (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(verts)
                a = majorAxisLength / 2
                b = minorAxisLength / 2
                circularity = round(np.sqrt(pow(a, 2) - pow(b, 2)) / a, 2)
                # print("Roundness",circularity)
                ################################ cell region properties calculation end ###############

                """ thresholding mean intensity value  to calculate the cell liveliness/viability """

                # if cell_state > 1:
                #     cell_state = 1
                # if cell_state < 0:
                #     cell_state = 0

                # deadThreshold = 90
                # precent20 = 115
                # precent30 = 125
                # precent40 = 135
                # precent50 = 145
                # precent60 = 150
                # precent80 = 160
                # precent90=170
                # precent100 = 180

                # if mean_val<deadThreshold:
                #     cell_state =0
                # elif deadThreshold<=mean_val<precent20:
                #     cell_state =10
                # elif precent20<=mean_val<precent30:
                #     cell_state =20
                # elif precent30<=mean_val<precent40:
                #     cell_state =30
                # elif precent40<=mean_val<precent50:
                #     cell_state =40
                # elif precent50<=mean_val<precent60:
                #     cell_state =50
                # elif precent60<=mean_val<precent80:
                #     cell_state =60
                # elif precent80<=mean_val<precent90:
                #     cell_state =80
                # elif precent80<=mean_val<precent100:
                #     cell_state =90
                # else:
                #     cell_state =100

                # TODO change max dead 60 pct to dynamically daluclated one from backgorudn
                cell_state = ((60 - np.clip((bkgrn - 15 - mean_val), 0, 60)) / 60) * 100



                masked_image_out = cv2.putText(masked_image_out, str(cell_state.round(2)), (int(x + 10), int(y + 20)),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                masked_image_out = cv2.putText(masked_image_out, str(mean_val), (int(x + 20), int(y + 40)),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cell_state_list.append(cell_state)
                circularity_list.append(circularity)
    masked_image_out = masked_image_out.astype(np.uint8)
    cv2.imwrite(filename, masked_image_out)

    # print(cell_state_list)

    return cell, cell_state_list, circularity_list, contour_area_list, bkgrn, bkgrnd_pix_analyzed
