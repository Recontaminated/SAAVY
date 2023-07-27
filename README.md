# Segmentation Algorithm to Assess the ViabilitY of 3D spheroid slices (aka SAAVY)

SAAVY was created for the purpose of predicting the viability percentage of 3D tissue cultures, cystic spheroids specifcally, according to brightfield micrscoppy plane slice images. SAAVY uses Mask R-CNN for instance segmentation to detect the spheroids in the focal plane within a provided image. Morphology is taken into account through our training with both live and dead spheroids. Live spheroids are distinctly spherical with noticeable edges whereas dead spheroids have a jagged outline from the apototic cell death. We based the viability algorithm on human expert assessmet and measure the intensity of the spheroid as compared to the background. Spheroids that have higher viabilities are typically closer in intensity values to that of the background, on average. Further, we include artificial noise in the backgrounds of the images to increase SAAVY's tolerance in the case of noisy biological backgrounds (i.e. matrix protein deposits, matrices loaded with foreign materials, and/or co-cultured cells creating a background).

SAAVY outputs the viability percent, average spheroid size, total count of spheorids included in the analysis, the total percent area of the image analyzed, and the average intensity value for the background. Our current code outputs the averages of each image, but maintains the ability to output specific viabilities, sizes, and intensity values for each invidivual spheroid identified in a given image.  

Installation/use:


For our specific use case, please see the preprint here:


Annotation: Annotate(polygon mask) the files using makesenseai tool and download the annottaion as vgg json format

Credits: https://github.com/miki998/Custom_Train_MaskRCNN/blob/master/train.py as a template for train.py and https://github.com/matterport/Mask_RCNN for the implementation of MRCNN!

Example detections (specks in background are nanoparticles): ![Pancreatic cancer organoids highlighted using MRCNN](https://raw.githubusercontent.com/Recontaminated/OrganoidSegmentation/master/mrcnn_training/main/Mask_RCNN/out/20210702_plate1_gem_c0.5_0001..jpg)

Example analysis (batch job) https://github.com/Recontaminated/OrganoidSegmentation/blob/master/mrcnn_training/main/Mask_RCNN/Analysis_output.csv
