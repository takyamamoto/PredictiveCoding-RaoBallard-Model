# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import network
from tqdm import tqdm

np.random.seed(0)

def DoG(img, ksize=(5,5), sigma=1.3, k=1.6):
    # DoG filter as a model of LGN
    g1 = cv2.GaussianBlur(img, ksize, sigma)
    g2 = cv2.GaussianBlur(img, ksize, k*sigma)
    dog = g1 - g2
    return (dog - dog.min())/(dog.max()-dog.min())

def GaussianMask(sizex=16, sizey=16, sigma=5):
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)
    
    x0 = sizex // 2
    y0 = sizey // 2
    mask = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*(sigma**2)))
    return mask / np.sum(mask)

# Preprocess of inputs
num_images = 5
num_iter = 4000

imgdirpath = "./images_preprocessed/"
imglist = []

for i in range(num_images):    
    filepath = imgdirpath + "{0:03d}.jpg".format(i + 1)
    #img_loaded = cv2.imread(filepath)[:, :, 0]
    img_loaded = cv2.imread(filepath)[:, :, 0].astype(np.float32)
    img_loaded /= 255
    img_loaded = DoG(img_loaded) 
    #img_loaded = (img_loaded-np.mean(img_loaded)) / np.std(img_loaded) 
    imglist.append(img_loaded)


# Define model
model = network.RaoBallard1999Model()

# Get image from imglist
error_list = []

# Simulation
#pbar = tqdm(total=num_iter)
H, W = imglist[0].shape
nt = 15
gmask = GaussianMask()
for iter_ in tqdm(range(num_iter)):
    i = np.random.randint(0, num_images)
    img = imglist[i]
    # Get the coordinates of the upper left corner of clopping image randomly.
    beginx = np.random.randint(0, W-27)
    beginy = np.random.randint(0, H-17)
    img_clopped = img[beginy:beginy+16, beginx:beginx+26]

    # Clop three inputs
    #inputs = np.array([(img_clopped[:, i*5:i*5+16]).flatten() for i in range(3)])
    inputs = np.array([(gmask*img_clopped[:, i*5:i*5+16]).flatten() for i in range(3)])
    
    inputs = (inputs - np.mean(inputs)) / np.std(inputs)
    #inputs = (((inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))) -0.5)*1.5
    
    # Reset states
    model.initialize_states()
    
    # Input an image patch until latent variables are converged 
    total_error = 0
    for _ in range(nt):
        error, errorh, r = model(inputs)
        total_error += model.calculate_total_error(error, errorh)
        
    error_list.append(total_error/nt)

    # Decay learning rate         
    if iter_ % 40 == 39:
        model.k2 /= 1.015
        
    if iter_ % 1000 == 0:        
        print("\n iter"+str(iter_)+":", error_list[iter_])
    
    #pbar.update(1)

# Plot results
plt.figure()
plt.plot(np.arange(num_iter), np.array(error_list)) 
plt.savefig("error.png")
plt.show()

# Plot Receptive fields of level 1
fig = plt.figure(figsize=(10, 5))
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(np.reshape(model.U[:, i], (16, 16)), cmap="gray")
    plt.axis("off")

plt.tight_layout()
fig.suptitle("Receptive fields of level 1", fontsize=20)
plt.subplots_adjust(top=0.9)
plt.savefig("RF_level1.png")
plt.show()

# Plot Receptive fields of level 2
"""
#U_ = np.zeros((416, 96)) # 416 = 16 x 26
#for i in range(3):
#    U_[i*80:i*80+256, i*32:(i+1)*32] = model.U
#Uh_rf = U_ @ model.Uh
#Uh_rf = np.reshape(Uh_, (16, 26, -1))
"""

Uh_rf = np.zeros((16, 26, 128))
Uh_ = np.array([model.U @ model.Uh[i*32:(i+1)*32] for i in range(3)])
Uh_ = np.reshape(Uh_, (3, 16, 16, -1))
for i in range(3):    
    Uh_rf[:, i*5:i*5+16] += Uh_[i]

#Uh_rf[:, 5:10] /= 2
#Uh_rf[:, 10:16] /= 3
#Uh_rf[:, 16:21] /= 2

fig = plt.figure(figsize=(10, 5))
for i in range(24):
    plt.subplot(4, 6, i+1)
    #plt.imshow(np.reshape(Uh_rf[:, i], (16, 26)), cmap="gray")
    plt.imshow(Uh_rf[:, :, i], cmap="gray")
    plt.axis("off")

plt.tight_layout()
fig.suptitle("Receptive fields of level 2", fontsize=20)
plt.subplots_adjust(top=0.9)
plt.savefig("RF_level2.png")
plt.show()
