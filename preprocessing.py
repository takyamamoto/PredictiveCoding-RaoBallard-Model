# -*- coding: utf-8 -*-

import cv2
import glob
import os 

def DoG(img, size=5, sigma=1, k=1.6, gamma=1):
    # DoG filter as a model of LGN
    g1 = cv2.GaussianBlur(img, (size, size), sigma)
    g2 = cv2.GaussianBlur(img, (size, size), sigma*k)
    return g1 - gamma*g2

imgdirpath = "./images"
imglist = glob.glob(imgdirpath+"/*")

outdirpath = "./images_preprocessed/"
os.makedirs(outdirpath, exist_ok=True)

for i in range(len(imglist)):    
    filepath = imglist[i]
    img_bgr = cv2.imread(filepath)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # img_gray = DoG(img_gray)
    cv2.imwrite(outdirpath + "{0:03d}.jpg".format(i + 1), img_gray)
