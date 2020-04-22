# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import network
from tqdm import tqdm

# Preprocess of inputs
num_images = 10
num_sampling = 200
num_iter = num_images * num_sampling

imgdirpath = "./images_preprocessed/"
imglist = []
for i in range(num_images):    
    filepath = imgdirpath + "{0:03d}.jpg".format(i + 1)
    img_loaded = cv2.imread(filepath)[:, :, 0].astype(np.float32)
    img_loaded /= 256
    img_loaded = (img_loaded - np.mean(img_loaded)) / np.std(img_loaded)
    imglist.append(img_loaded)


# Define model
model = network.RaoBallard1999Model()

# Get image from imglist
error_list = []

# Simulation
pbar = tqdm(total=num_iter)
for i in range(num_images):    
    img = imglist[i] 
    H, W = img.shape
    
    for j in range(num_sampling):
        # Get the coordinates of the upper left corner of clopping image randomly.
        beginx = np.random.randint(0, W-27)
        beginy = np.random.randint(0, H-17)
        img_clopped = img[beginy:beginy+16, beginx:beginx+26]

        # Clop three inputs
        inputs = np.array([img_clopped[:, 0:16].flatten(), 
                           img_clopped[:, 5:21].flatten(), 
                           img_clopped[:, 10:26].flatten()])

        for _ in range(2000):    
            error = model(inputs)
        
        error_list.append(np.mean(error**2))
        
        if j % 40 == 0:
            model.k2 *= 0.985
        
        pbar.update(1)

plt.figure()
plt.plot(np.arange(num_iter), np.array(error_list)) 
plt.show()

plt.figure(figsize=(10, 5))
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(np.reshape(model.U[0, :, i], (16, 16)), cmap="gray")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()