import cv2
import os

path = "./raw_data"
os.makedirs(f"{path}_resized", exist_ok=True)
for file in os.listdir(path):
    f = os.path.join(path, file) 
    print(f"Resizing {f}")
    img = cv2.imread(f)
    resized = cv2.resize(img,(1360,1024), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(img,None,fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    newName = "./raw_data_resized/" + file[:-4] + ".jpg"
    print(newName)
    cv2.imwrite(newName, resized)
    



