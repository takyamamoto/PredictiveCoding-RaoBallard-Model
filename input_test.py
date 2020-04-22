# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
inputs = [img_clopped[:, 0:16], 
          img_clopped[:, 5:21], 
          img_clopped[:, 10:26]]

# Show clopped images
plt.figure(figsize=(5,10))
ax1 = plt.subplot(1,2,1)
plt.title("Orignal image")
plt.imshow(img, cmap="gray")
ax1.add_patch(patches.Rectangle(xy=(beginx, beginy), 
                                width=26, height=16, ec='red', fill=False))

ax2 = plt.subplot(1,2,2)
plt.title("Clopped image")
plt.imshow(img_clopped, cmap="gray")
plt.tight_layout()
plt.show()


# Define model
# model = network.RaoBallard1999Model()

# Simulation