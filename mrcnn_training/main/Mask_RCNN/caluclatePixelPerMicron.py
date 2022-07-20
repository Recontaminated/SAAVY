from tkinter import *
import tkinter
from tkinter import filedialog
from turtle import pos
import cv2
import math
root = Tk()
root.withdraw()
somefile = filedialog.askopenfilename()

image = cv2.imread(somefile)
width, height, channels = image.shape
print(f"{width}x{height}")
pos1 =()
pos2 =()
def click_event(event, x, y, flags, params):
    global pos1, pos2
    font = cv2.FONT_HERSHEY_SIMPLEX
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        pos1 = (x,y)
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window

        cv2.circle(image, (x,y), 5, (235,0,0), thickness=10, lineType=8, shift=0)
        cv2.putText(image, "Pos 1",
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', image)
        
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
        pos2 = (x,y)
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        cv2.circle(image, (x,y), 5, (0,0,235), thickness=10, lineType=8, shift=0)
        cv2.putText(image, "Pos 2",
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', image)
        
cv2.imshow('image', image)

# setting mouse handler for the image
# and calling the click_event() function
cv2.setMouseCallback('image', click_event)

# wait for a key to be pressed to exit
cv2.waitKey(0)
print(pos1)
pos1A,pos1B = pos1
pos2A,pos2B = pos2
dist = math.sqrt((pos1A-pos2A)**2 + (pos1B-pos2B)**2)
print(dist)
refrenceUM = input("Enter the refrence um: ")

pixelsPerum = dist/float(refrenceUM)
print(pixelsPerum)
# close the window
cv2.destroyAllWindows()