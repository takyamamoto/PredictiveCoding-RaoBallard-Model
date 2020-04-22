# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import network
"""
# Preprocess of inputs
imgdirpath = "./images_preprocessed/"

imglist = []
for i in range(10):    
    filepath = imgdirpath + "{0:03d}.jpg".format(i + 1)
    img_loaded = cv2.imread(filepath)[:, :, 0]
    imglist.append(img_loaded)

# Get image from imglist
img = imglist[0]
H, W = img.shape

# Get the coordinates of the upper left corner of clopping image randomly.
beginx = np.random.randint(0, W-1)
beginy = np.random.randint(0, H-1)
img_clopped = img[beginy:beginy+16, beginx:beginx+26]

# Clop three inputs
inputs = np.array([img_clopped[:, 0:16].flatten(), 
                   img_clopped[:, 5:21].flatten(), 
                   img_clopped[:, 10:26].flatten()])
"""
# Define model
model = network.RaoBallard1999Model()

# Simulation