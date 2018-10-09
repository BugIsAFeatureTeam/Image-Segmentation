# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 21:48:53 2018

@author: 58011256
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from skimage.io import imread , imshow
from skimage import color
from skimage import measure
import math

img = cv2.imread('1.png')

for i in range(img.shape[0]):
    for j in range(0,2):
        img[i,j,:] = 0
        img[j,i,:] = 0
        last_idx = img.shape[1]-1-j
        img[i,last_idx,:] = 0
        img[last_idx,i,:] = 0