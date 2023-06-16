
import os
import sys
p = os.path.abspath('../..')
if p not in sys.path:
    sys.path.append(p)
from PIL import Image

from pytorch import perdict as segment


from cell_perdict import get_prediction as get_cell_prediction
import os
import cv2



DATADIR= r"../trainingData"

count = 0
for img in os.listdir(DATADIR):
    #check if image is a png or jpg
    if img[-3:] != "png" and img[-3:] != "jpg":
        continue
    img_path = os.path.join(DATADIR, img)
    confidence_thresh = 0.5
    masks, boxes, pred_cls, confidence_scores = segment.get_prediction(img_path, confidence_thresh)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(masks)):
        pt1 = tuple(map(int,boxes[i][0]))
        pt2 = tuple(map(int,boxes[i][1]))

        # crop image with padding
        padding = 10
        crop_img = img[pt1[1]-padding:pt2[1]+padding, pt1[0]-padding:pt2[0]+padding]
        #show image
        cv2.imshow("cropped", crop_img)
        print("viability is: ", get_cell_prediction(Image.fromarray(crop_img)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    




