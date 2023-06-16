from perdict import get_prediction
import os
import cv2

# make dataset directory
def make_dataset_dir():
    if not os.path.exists('cell_model/dataset'):
        os.makedirs('cell_model/dataset')

DATADIR= "trainingData"
make_dataset_dir()

count = 0
for img in os.listdir(DATADIR):
    img_path = os.path.join(DATADIR, img)
    confidence_thresh = 0.5
    masks, boxes, pred_cls, confidence_scores = get_prediction(img_path, confidence_thresh)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(masks)):
        pt1 = tuple(map(int,boxes[i][0]))
        pt2 = tuple(map(int,boxes[i][1]))

        # crop image with padding
        padding = 10
        crop_img = img[pt1[1]-padding:pt2[1]+padding, pt1[0]-padding:pt2[0]+padding]
        #show image
        print("Viability? 1-9...(1)0")
        cv2.imshow("cropped", crop_img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if 48 <= key <= 57:  # Check if the key is a number (0-9)
                classification = key - 48
                print(f"Classified as: {classification}")
                break
        cv2.destroyAllWindows()
        if classification == 0:
            classification = 10

        viability = (classification) *10
        # save image
        print("saving image")
        cv2.imwrite("cell_model/dataset/" + str(count) + ".png", crop_img)

        # append viability
        with open("cell_model/dataset/"+"data.txt", "a") as f:
            f.write(str(viability) + "\n")
        count += 1



