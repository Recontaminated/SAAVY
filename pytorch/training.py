#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torch
import torch.utils.data
import json
import os
import PIL
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("training", help="training folder")
parser.add_argument("validation", help="validation folder")
args = parser.parse_args()
# import scripts.pytorchVisionScripts.utils as utils
# from scripts.pytorchVisionScripts.engine import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


import torchvision.transforms.functional as TF
from torchScripts import utils
from torchScripts.engine import train_one_epoch, evaluate

# In[2]:


class OrganoidDataset(torch.utils.data.Dataset):
    """
    flow curtosy of pytorch.org's finetuning documentation
    """

    def loadMasks(self, root):
        masks = {}
        fsImages = os.listdir(os.path.join(root))
        with open(os.path.join(root, self.via)) as f:
            data = json.load(f)
            for key in data:
                if data[key]["filename"] in fsImages:
                    # check if regions exist is empty and if so remove the image from the list
                    # TODO: make it so that null images can be used for training
                    if data[key]["regions"] == []:
                        # self.imgs.remove(data[key]["filename"])
                        pass
                    else:
                        masks[data[key]["filename"]] = data[key]["regions"]

        return masks

    def __init__(self, root, via, shouldtransforms=False):
        self.root = root
        self.via = via
        self.shouldtransforms = shouldtransforms
        # load all image files, sorting them to
        # ensure that they are aligned
        files = os.listdir(os.path.join(root))
        # ignrore all .json files
        self.masks = self.loadMasks(root)
        self.imgs = list(self.masks.keys())

    def transform(self, image, mask):
        # # Resize
        # resize = transforms.Resize(size=(520, 520))
        # image = resize(image)
        # mask = resize(mask)

        # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, idx):
        imagePath = os.path.join(self.root, self.imgs[idx])
        img = PIL.Image.open(imagePath).convert("RGB")
        mask = self.masks[self.imgs[idx]]
        # mask is a dictionary of all x points and all y points. we have to convert teese to a binary mask
        if self.shouldtransforms:
            img, target = self.transform(img, mask)
        masks = []
        for key in mask:
            points = key["shape_attributes"]
            x = points["all_points_x"]
            y = points["all_points_y"]
            # we can make a binary mask from this, but we need to know the size of the image
            # we can get this from the image itself
            width, height = img.size
            mask = PIL.Image.new("L", (width, height), 0)
            PIL.ImageDraw.Draw(mask).polygon(
                list(zip(x, y)), outline=1, fill=1
            )  # not sure if this is efficeint lol
            mask = np.array(mask, dtype=bool)
            masks.append(mask)
        numObjs = len(masks)
        boxes = []

        for i in range(numObjs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class, why?
        # we can use the number of masks as the number of labels
        labels = torch.ones((numObjs,), dtype=torch.int64)
        masks = np.array(masks, dtype=np.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        iscrowd = torch.zeros(
            (numObjs,), dtype=torch.int64
        )  # we need this because cocodataset has crowd (single instance) to be zero
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd
        target["area"] = area
        img = TF.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def buildModel(numClasses):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        inFeatures = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(inFeatures, numClasses)
        inFeaturesMask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hiddenLayer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            inFeaturesMask, hiddenLayer, numClasses
        )
        return model

    # def getTransform(train):
    #     transforms = []
    #     transforms.append(T.ToTensor())
    #     if train:
    #         #randomly flip the image and ground truth for data augmentation
    #         transforms.append(T.RandomHorizontalFlip(0.5))
    #     return T.Compose(transforms)

    # split the dataset in train and test set
    def trainTestSplit():
        folder = "trainingData"
        # check if trainingData/train and trainingData/test exist
        if not os.path.exists(os.path.join(folder, "train")):
            os.mkdir(os.path.join(folder, "train"))

        if not os.path.exists(os.path.join(folder, "test")):
            os.mkdir(os.path.join(folder, "test"))

        # copy 10% of the images to the test folder
        for file in os.listdir(os.path.join(folder, "images")):
            if np.random.rand(1) < 0.1:
                os.rename(
                    os.path.join(folder, "images", file),
                    os.path.join(folder, "test", file),
                )
                os.rename(
                    os.path.join(folder, "via_region_data.json"),
                    os.path.join(folder, "test", "via_region_data.json"),
                )
            else:
                os.rename(
                    os.path.join(folder, "images", file),
                    os.path.join(folder, "train", file),
                )
                os.rename(
                    os.path.join(folder, "via_region_data.json"),
                    os.path.join(folder, "train", "via_region_data.json"),
                )

    # we have a train test split so we dont need to do this

    dataset = OrganoidDataset(args.training, "via_region_data.json", False)
    validationDataset = OrganoidDataset(args.validation, "", False)
    dataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
    )
    validationDataLoader = torch.utils.data.DataLoader(
        validationDataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    num_classes = 2

    model = buildModel(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 15

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations

        train_one_epoch(
            model, optimizer, dataLoader, device, epoch, writer, print_freq=1
        )
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, validationDataLoader, writer, epoch, device=device)

    torch.save(model, "torchDebug01.1.pt")


# if __name__ == "__main__":
main()
writer.flush()
