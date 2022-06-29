
import cv2
import os
import sys
from cv2 import mean
from cv2 import normalize
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
    #Apply the given mask to the image.
    
   
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks, class_ids, scores, threshold, filename,Debug=False):
#def display_instances(image, boxes, masks, class_ids, filename, threshold, scores=None):
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
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig_img = image.astype(np.uint8)
    masked_image = image.astype(np.uint32).copy()
    #dead = 0
    #live = 0
    cell = 0
    colors = [[0,1, 0]]
        
    cell_state_list =[]
    circularity_list =[]
    for i in range(N):
	
        if scores[i] > threshold :  
            color = colors[class_ids[i] - 1]
            if class_ids[i] - 1 == 0:
                cell = cell + 1
            #elif class_ids[i] - 1 == 1:
                #dead = dead + 1

            # Apply mask for each detected object
            mask = masks[:, :, i]
            #print("################################")
            #mask1 = 1*mask
            #print(mask1)S
            #cv2.imwrite("mask.jpg",mask1)
        
            masked_image = apply_mask(masked_image, mask, color)
            
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
       
            padded_mask[1:-1, 1:-1] = mask
            
            contours = find_contours(padded_mask, 0.5)
            #print(contours)
            masked_image = masked_image.astype(np.uint8)       
            thickness = 2
            
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)           
                verts = np.fliplr(verts) - 1
                verts = np.array([verts], np.int32)  
                verts = verts.reshape((-1,1,2))
                clr = (color[0] * 250)
                clr = [0,250, 0]
                masked_image_out = cv2.polylines(masked_image,[verts],True,clr,thickness)               
                mask= np.zeros(grey_img.shape,np.uint8)
                cv2.drawContours(mask,[verts],0,255,-1)
                x,y,w,h = cv2.boundingRect(verts) # offsets - with this you get 'mask'              
                crop_img = orig_img[y:y+h,x:x+w]
                #cv2.imshow('cutted contour',crop_img)
                #cv2.waitKey()
             
                ################################## cell region properties calculation start ##########
               #reference: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
                avg_pixel_cell = np.mean(crop_img)
                #print("avg_pixel_cell crop bb:",avg_pixel_cell)
                
                contour_area = cv2.contourArea(verts)
                #print("contour area: ",contour_area)
                
                rect_area = w*h
                #print("rect_area:",rect_area)
                
                extent = float(contour_area)/rect_area  #object area/bounding rectangle area
                #print("extent: ",extent)
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grey_img,mask = mask)
                #print("min_max values:",min_val,max_val)
                """ Average intensity value of a cell """
                mean_val = cv2.mean(grey_img,mask = mask)
                mean_val =list(mean_val)
                mean_val = mean_val[0]
                #print("mean_intensity: ",mean_val)
                
                """eccentricity/circularity of a circle is 0"""
                (x,y), (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(verts)
                a = majorAxisLength / 2
                b = minorAxisLength / 2
                circularity = round(np.sqrt(pow(a, 2) - pow(b, 2))/a, 2)
                #print("Roundness",circularity)
                ################################ cell region properties calculation end ###############
                
                """ thresholding mean intensity value  to calculate the cell liveliness/viability """
                # endpoints = (50,175)
                # minval, maxval = endpoints

                # normalize = (mean_val - minval) / (maxval - minval)
                # cell_state = normalize

                deadThreshold = 80
                precent20 = 100
                precent50 = 120
                precent80 = 130
                precent100 = 140
                


                print(mean_val)
                if 0<mean_val<=deadThreshold:
                    cell_state = "0% live"
                    cell_state =0
                    #print(cell_state)
                
                elif deadThreshold<mean_val<=precent20:
                    cell_state =1
                elif precent20<mean_val<=precent50:
                    cell_state =20
                elif precent50<mean_val<=precent80:
                    cell_state =50
                elif precent80<mean_val<=precent100:
                    cell_state =80
                elif mean_val>precent100:
                    cell_state =100
                
                    
                # elif 80<mean_val<99:
                #     cell_state = "20% live"
                #     cell_state =20
                #     #print(cell_state)
                     
                # elif 100<mean_val<129:
                #     cell_state = "50% live"
                #     cell_state =50
                #     #print(cell_state)
                    
                # elif 130<mean_val<140:
                #     cell_state = "80%live"
                #     cell_state =80
                #     #print(cell_state)
                    
                # elif 150<mean_val<159:
                #     cell_state = "90%live"
                #     cell_state =90
                #     #print(cell_state)
                    
                # elif 160<mean_val<256:
                #     cell_state = "100%live"
                #     cell_state =100
                #     #print(cell_state)
                
                
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                if Debug:
                    print("mean val for file: "+ filename +"is: "+ str(mean_val))
                    cv2.imshow('single cell',crop_img)
                    cv2.waitKey()
                    cv2.imshow('masked_image',mask)
                    cv2.waitKey()
                    cv2.imshow('masked_image_out',masked_image_out)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
        
            masked_image_out = masked_image_out.astype(np.uint8)
            cv2.imwrite(filename, masked_image_out)
        cell_state_list.append(cell_state)
        circularity_list.append(circularity)
       # print(cell_state_list)

	
    return cell,cell_state_list,circularity_list





