# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:40:26 2018

@author: 58011256
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage
from skimage.io import imread , imshow, imsave
from skimage import color
from skimage import data
from skimage import measure
from skimage.filters import threshold_mean
from skimage.measure import label, regionprops
from skimage.draw import polygon
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from skimage.io import imread , imshow
from skimage import color
from skimage import measure
from skimage import exposure
import math
from skimage.morphology import opening, binary_opening
from skimage.morphology import disk
import os
from sklearn.cluster import KMeans
import math

PATH = 'segmentation_WBC-master/'
all_files = next(os.walk(PATH+'Dataset 1'))[2]
img = color.rgb2hsv(imread(PATH+'Dataset 1/'+'003.bmp'))
class DominantColors:

    CLUSTERS = None
    IMAGE = None
    TIMG = None
    COLORS = None
    LABELS = None
    ORIG = None
    WCSS = None
    pLABEL = None
    
    def __init__(self, image):
        #read image
        img = cv2.imread(image)
        
        #convert to rgb from bgr
        #img = color.rgb2hsv(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        self.ORIG = img.copy()
        
        # Contrast stretching
        p2, p98 = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(p2, p98))
        self.TIMG = img.copy()
        
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 1))
        
        #save image after operations
        self.IMAGE = img
        
    def dominantColors(self):
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS, init = 'k-means++', random_state = 0)
        kmeans.fit(self.IMAGE)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #find label of purple color
        #self.pLABEL = kmeans.predict([[128,0,128]])
        #self.pLABEL = kmeans.predict([[150,255,127]])
        self.pLABEL = kmeans.predict([[0]])
        
        #returning after converting to integer from float
        return self.COLORS
    
    def setIMG(self,img):
        # Contrast stretching
        p2, p98 = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(p2, p98))
        self.TIMG = img.copy()
        
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 1))
        
        #save image after operations
        self.IMAGE = img
    
    def showO(self):
        imshow(self.ORIG)
    
    def showTIMG(self):
        imshow(self.TIMG)
        
    def elbow(self):
        #Elbow method
        wcss = []
        for i in range(1, 10):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(self.IMAGE)
            wcss.append(kmeans.inertia_)
        self.WCSS = wcss
        
    def findK(self):
        m = (self.WCSS[0]-self.WCSS[8])/(1-9)
        c = self.WCSS[0]-m*1
        A=m
        B=-1
        maxD = 0
        for x1,y1 in enumerate(self.WCSS,1):
            d = abs(A*x1+B*y1+c)/math.sqrt(A**2+B**2)
            if d > maxD:
                maxD = d
                self.CLUSTERS=x1+1
        print('found :'+str(self.CLUSTERS)+' group')
        return self.CLUSTERS

def K_sep():
    img = 'segmentation_WBC-master/Dataset 1/'+file
    dc = DominantColors(img) 
    dc.elbow()
    dc.findK()
    dc.dominantColors()
    new = np.reshape(dc.LABELS, (-1, dc.ORIG.shape[1]))
    for i in range(0,new.shape[0]):
        for j in range(0,new.shape[1]):
            if new[i][j] == dc.pLABEL[0]:
                new[i][j] = 255
            else:
                new[i][j] = 0
    Gopened = opening(new)
    Nsub = dc.ORIG.copy()
    for i in range (0,Gopened.shape[0]):
        for j in range (0,Gopened.shape[1]):
            if Gopened[i][j] >250:
                Nsub[i][j] = 255
    dc.setIMG(Nsub) 
    dc.elbow()
    dc.findK()
    dc.dominantColors()
    new = np.reshape(dc.LABELS, (-1, dc.ORIG.shape[1]))
    for i in range(0,new.shape[0]):
        for j in range(0,new.shape[1]):
            if new[i][j] == dc.pLABEL[0]:
                new[i][j] = 70
            else:
                new[i][j] = 0
    Copened = opening(new)
    last = Gopened+Copened
    imsave(PATH+'/Granule+Cytoplasm/'+file+'.png',last)

for file in all_files:
    if file.find('.bmp') > 0:
        #contour_sep()
        K_sep()