# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import network
from tqdm import tqdm

def DoG(img, ksize=(5,5), sigma=1.3, k=1.6):
    # DoG filter as a model of LGN
    g1 = cv2.GaussianBlur(img, ksize, sigma)
    g2 = cv2.GaussianBlur(img, ksize, k*sigma)
    dog = g1 - g2
    return (dog - dog.min())/(dog.max()-dog.min())

def GaussianMask(sizex=16, sizey=16, sigma=4.8):
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)
    
    x0 = sizex // 2
    y0 = sizey // 2
    mask = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*(sigma**2)))
    return mask / np.sum(mask)

# Preprocess of inputs
num_images = 10
#num_sampling = 
#num_iter = num_images * num_sampling
num_iter = 10000

imgdirpath = "./images_preprocessed/"
imglist = []

for i in range(num_images):    
    filepath = imgdirpath + "{0:03d}.jpg".format(i + 1)
    img_loaded = cv2.imread(filepath)[:, :, 0].astype(np.float32)
    img_loaded /= 255
    img_loaded = DoG(img_loaded)
    #img_loaded = cv2.GaussianBlur(img_loaded, (5,5), 1.3)
    #img_loaded = (img_loaded - np.mean(img_loaded)) / np.std(img_loaded)
    imglist.append(img_loaded)

# Define model
model = network.RaoBallard1999Model()

# Get image from imglist
error_list = []

# Simulation
#pbar = tqdm(total=num_iter)
H, W = imglist[0].shape

gmask = GaussianMask()
for j in tqdm(range(num_iter)):
    i = np.random.randint(0, num_images)
    img = imglist[i]
    # Get the coordinates of the upper left corner of clopping image randomly.
    beginx = np.random.randint(0, W-27)
    beginy = np.random.randint(0, H-17)
    img_clopped = img[beginy:beginy+16, beginx:beginx+26]

    # Clop three inputs
    inputs = np.array([(gmask*img_clopped[:, 0:16]).flatten(), 
                       (gmask*img_clopped[:, 5:21]).flatten(), 
                       (gmask*img_clopped[:, 10:26]).flatten()])
    inputs = (inputs - np.mean(inputs)) / np.std(inputs)
    
    # Reset states
    model.initialize_states()
    #rtm1 = 0
    
    # Input an image patch until latent variables are converged 
    for _ in range(30):
        error, errorh, r = model(inputs)

    total_error = model.calculate_total_error(error, errorh)
    error_list.append(total_error)

    # Decay learning rate         
    if j % 40 == 39:
        model.k2 /= 1.015
        
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
    plt.imshow(np.reshape(model.U[:, i], (16, 16)), cmap="gray")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.savefig("RF.png")
plt.show()
