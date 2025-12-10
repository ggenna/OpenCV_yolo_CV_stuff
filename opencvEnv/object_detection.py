# Import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

%matplotlib inline


# Deep Learning based Object Detection
# Satya Mallick, LearnOpenCV.com

# Architecture : Mobilenet based Single Shot Multi-Box (SSD)
# Framework : Tensorflow

# You have 2 options here:

# Automatic setup: By runnning the code cells below all the necessay files will be downloaded at once and will be ready to use.
# Manual Setup: In this case, you'll have to download and perform the required setup manually.
# Instructions for Manual Setup
# Download Model files from Tensorflow model ZOO
# Model files can be downloaded from the Tensorflow Object Detection Model Zoo: tf2_detection_zoo.md

# Download mobilenet model file
# You can download the model TAR.GZ file and uncompress it.

# After Uncompressing and put the highlighed file (along with the folder) in a models folder.

# ssd_mobilenet_v2_coco_2018_03_29
# |─ checkpoint
# |─ frozen_inference_graph.pb
# |─ model.ckpt.data-00000-of-00001
# |─ model.ckpt.index
# |─ model.ckpt.meta
# |─ pipeline.config
# |─ saved_model
# |─── saved_model.pb
# |─── variables

# Create config file from frozen graph
# Extract the files
# Run the tf_text_graph_ssd.py file with input as the path to the frozen_graph.pb file and output as desired.
# A sample config file has been included in the models folder

# A Script to download and extract model tar.gz file.

# if not os.path.isdir('models'):
#     os.mkdir("models")

# if not os.path.isfile(modelFile):
#     os.chdir("models")
#     # Download the tensorflow Model
#     urllib.request.urlretrieve('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz', 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

#     # Uncompress the file
#     !tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

#     # Delete the tar.gz file
#     os.remove('ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

#     # Come back to the previous directory
#     os.chdir("..")
# The final directory structure should look like this:

# ├─── coco_class_labels.txt        
# ├─── tf_text_graph_ssd.py
# └─── models
#      ├───ssd_mobilenet_v2_coco_2018_03_29.pbtxt
#      └───ssd_mobilenet_v2_coco_2018_03_29
#          └───frozen_inference_graph.pb

