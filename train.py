# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import network
from tqdm import tqdm

# Preprocess of inputs
num_images = 10
#num_sampling = 
#num_iter = num_images * num_sampling
num_iter = 100000

imgdirpath = "./images_preprocessed/"
imglist = []
for i in range(num_images):    
    filepath = imgdirpath + "{0:03d}.jpg".format(i + 1)
    img_loaded = cv2.imread(filepath)[:, :, 0].astype(np.float32)
    img_loaded /= 255
    #img_loaded = (img_loaded - np.mean(img_loaded)) / np.std(img_loaded)
    imglist.append(img_loaded)


# Define model
model = network.RaoBallard1999Model()

# Get image from imglist
error_list = []

# Simulation
#pbar = tqdm(total=num_iter)
H, W = imglist[0].shape
"""
for i in range(num_images):
    img = imglist[i]     
"""
for j in tqdm(range(num_iter)):
    i = np.random.randint(0, num_images)
    img = imglist[i]
    # Get the coordinates of the upper left corner of clopping image randomly.
    beginx = np.random.randint(0, W-27)
    beginy = np.random.randint(0, H-17)
    img_clopped = img[beginy:beginy+16, beginx:beginx+26]

    # Clop three inputs
    inputs = np.array([img_clopped[:, 0:16].flatten(), 
                       img_clopped[:, 5:21].flatten(), 
                       img_clopped[:, 10:26].flatten()])
    
    # Reset states
    model.initialize_states()
    #rtm1 = 0
    
    # Input an image patch until latent variables are converged 
    for _ in range(15):
        error, errorh, r = model(inputs)

    """
    while True:
        error, errorh, r = model(inputs)
        diffr = np.mean(np.abs(r - rtm1))
        rtm1 = r.copy()
        if diffr < 1e-5:
            break
    """

    total_error = model.calculate_total_error(error, errorh)
    error_list.append(total_error)

    # Decay learning rate         
    if j % 40 == 0:
        model.k2 *= 0.985
        
    if j % 1000 == 0:        
        print("\n iter"+str(j)+":", total_error)
    
    #pbar.update(1)

# Plot results
plt.figure()
plt.plot(np.arange(num_iter), np.array(error_list)) 
plt.savefig("error.png")
plt.show()

plt.figure(figsize=(10, 5))
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(np.reshape(model.U[0, :, i], (16, 16)), cmap="gray")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.savefig("RF.png")
plt.show()
