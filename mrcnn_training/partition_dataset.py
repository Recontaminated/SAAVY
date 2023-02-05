
import os
import re
from shutil import copyfile
import math
import random


def iterate_dir(source, ratio):
    #print(source)
    #source= source.replace('\\', '/')
    #print(source)
    source = "raw_data_resized"
    #train_dir = os.path.join(source, 'train')
    #test_dir = os.path.join(source, 'test')
    train_dir ="./main/dataset/train"
    test_dir = "./main/dataset/val"
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]
    #images =[f for f in os.listdir(source)]

    num_images = len(images)
    print(num_images)
    num_test_images = math.ceil(ratio*num_images)
    print(num_test_images)

    for i in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        print(idx)
        filename = images[idx]
        copyfile(os.path.join(source, filename),
                 os.path.join(test_dir, filename))
       
        images.remove(images[idx])
        

    for filename in images:
        copyfile(os.path.join(source, filename),
                 os.path.join(train_dir, filename))
       

if __name__ == '__main__':
    imageDir = 'raw_data_resized'
    iterate_dir(imageDir, 0.2)

