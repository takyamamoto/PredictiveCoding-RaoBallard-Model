# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.random.seed(0)

def DoG(img, ksize=(5,5), sigma=1.3, k=1.6):
    # DoG filter as a model of LGN
    g1 = cv2.GaussianBlur(img, ksize, sigma)
    g2 = cv2.GaussianBlur(img, ksize, k*sigma)
    dog = g1 - g2
    return (dog - dog.min())/(dog.max()-dog.min())

# Preprocess of inputs
imgdirpath = "./images_preprocessed/"

imglist = []
for i in range(10):    
    filepath = imgdirpath + "{0:03d}.jpg".format(i + 1)
    img_loaded = cv2.imread(filepath)[:, :, 0].astype(np.float32)
    img_loaded = DoG(img_loaded)
    #img_loaded = cv2.GaussianBlur(img_loaded, (5,5), 1.3)
    imglist.append(img_loaded)

# Get image from imglist
img = imglist[0]
H, W = img.shape

# Get the coordinates of the upper left corner of clopping image randomly.
beginx = np.random.randint(0, W-27)
beginy = np.random.randint(0, H-17)
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

def gaussian_2D(x, y, sig):
    return ((1.22)/(2*np.pi*(sig**2))) * np.exp(-(x**2 + y**2)/(2*(sig**2)))

def Gauss2Dwindow(sigma=10):
    num = 16; x = np.arange(-(num-1)/2, (num+1)/2, 1); g_window = np.zeros((x.shape[0],x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            g_window[i, j] = gaussian_2D(x[i], x[j], sigma)
    return g_window

gauss = Gauss2Dwindow()
I0 = gauss*40*(img_clopped[:, 0:16])
I1 = gauss*40*(img_clopped[:, 5:21])
I2 = gauss*40*(img_clopped[:, 10:26])

plt.figure()
plt.subplot(1,2,1)
plt.imshow(I0)
plt.subplot(1,2,2)

plt.imshow(img_clopped[:, 0:16])
plt.show()
