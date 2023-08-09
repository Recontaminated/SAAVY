# SAAVY Pytorch



### Python requirements

* Python >= 3.9
* Pytorch >= 2.0
* Pillow >= 9.4.0
* mattplotlib >= 3.7.1
* (Optional but highly recomended) cuda-toolkit = 11.8 
* Conda installation


## Basic use
Clone this repository

```
git clone https://github.com/armanilab/SAAVY.git
cd SAAVY
```
Create and activate the conda environment

*this may be differnet if you do not have a CUDA GPU, if so, install the packages manually from PyTorch and the rquirements section*

```
conda env create -f env.yml 
conda activate torch

```
Run the analysis 
```
python perdict.py --input "YOUR FOLDER HERE" --output "CREATE A FOLDER HERE"
```

## Fine tune model
1. download the [VIA image annotator](https://images.duckarmada.com/3DmAPCM5k7xb)
2. load images into dataset and create polygon masks around them

![.](https://images.duckarmada.com/5Qw1y2DW2t4s/direct.png)

3. Export as JSON. **Name it**  `via_region_data.json`

![](https://images.duckarmada.com/Rmr7SCBEhTOX/direct.png)

4. Copy both JSON and annotated images into training directory, 

5. Repeat from step 2 for validation dataset

6. run `python training.py --training "training/" --validation "validation/"` 

7. model will be saved to working directory


