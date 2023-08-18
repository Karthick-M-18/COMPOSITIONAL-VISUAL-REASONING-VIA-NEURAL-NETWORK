# COMPOSITIONAL-VISUAL-REASONING-VIA-NEURAL-NETWORK
COMPOSITIONAL VISUAL REASONING VIA NEURAL NETWORK using CLEVR Dataset and Tensorflow 1.x
This is a replication study of the project https://github.com/stanfordnlp/mac-network

## Requirements
- The project is developed using Tensorflow 1.x hence the Tensorflow 2.x has to be disabled (This step is updated in the python files)
- I've have performed experiments on Tesla 4 GPU that comes with the pro version of Google colab
- Hence, No library installation is required.


# Imports in Google Colab
```bash
Import statements
import os
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from google.colab import drive
from google.colab import runtime
from PIL import Image
drive.mount('/content/drive')
```

# Data Pre-processing
downloading the CLEVR dataset and extracting features for the images just like mentioned in the original paper.

### Dataset
To download and unpack the data, run the following commands:
```bash
!rm -rf CLEVR_v1
!wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
!unzip CLEVR_v1.0.zip
!mv CLEVR_v1.0 CLEVR_v1
!mkdir CLEVR_v1/data
!mv CLEVR_v1/questions/* CLEVR_v1/data/
!rm CLEVR_v1.0.zip
```
### Feature Extraction
We use ResNet-101 model to extract features from the images
```bash
# Caution! this step takes a lot of time as we have 77 Gigabytes of data to be uploaded in the drive
%run /content/drive/MyDrive/CLEVR_v1.0/extract_features.py --input_image_dir /content/CLEVR_v1/images/train --output_h5_file /content/drive/MyDrive/CLEVR_v1.0/CLEVR_v1/data/train.h5 --batch_size 32
%run /content/drive/MyDrive/CLEVR_v1.0/extract_features.py --input_image_dir /content/CLEVR_v1/images/val --output_h5_file /content/drive/MyDrive/CLEVR_v1.0/CLEVR_v1/data/val.h5 --batch_size 32
%run /content/drive/MyDrive/CLEVR_v1.0/extract_features.py --input_image_dir /content/CLEVR_v1/images/test --output_h5_file /content/drive/MyDrive/CLEVR_v1.0/CLEVR_v1/data/test.h5 --batch_size 32
```
### Gif
![](https://github.com/Karthick-M-18/COMPOSITIONAL-VISUAL-REASONING-VIA-NEURAL-NETWORK/blob/main/Gif1.gif)
![](https://github.com/Karthick-M-18/COMPOSITIONAL-VISUAL-REASONING-VIA-NEURAL-NETWORK/blob/main/Gif2.gif)

### Please refer the file COMPOSITIONAL_VISUAL_REASONING_VIA_NEURAL_NETWORK.ipunb for the entire code
