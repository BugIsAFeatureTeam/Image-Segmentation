# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:03:05 2018

@author: 58011256
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from skimage.io import imread , imshow, imsave
from skimage import color
from skimage import measure
import math
import os
import numpy as np
from skimage.morphology import square,disk,diamond,remove_small_objects,remove_small_holes
from skimage.morphology import erosion, dilation, opening,closing

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    ORIG = None
    WCSS = None
    pLABEL = None
    KMEAN = None
    
    def __init__(self, image):
        self.IMAGE = image
        
    def dominantColors(self):
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
        kmeans.fit(self.IMAGE)
        
        self.KMEAN = kmeans
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #find label of purple color
        #self.pLABEL = kmeans.predict([[128,0,128]])
        self.pLABEL = kmeans.predict([[150,255,127]])
        
        #returning after converting to integer from float
        return self.COLORS
    
    def show(self):
        imshow(self.ORIG)
        
    def elbow(self):
        #read image
        img = cv2.imread(self.IMAGE)
        
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
        
        for i in range(img.shape[0]):
            for j in range(0,2):
                img[i,j,:] = 0
                img[j,i,:] = 0
                last_idx = img.shape[1]-1-j
                img[i,last_idx,:] = 0
                img[last_idx,i,:] = 0

        self.ORIG = img.copy()
           
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #Elbow method
#        wcss = []
#        for i in range(1, 10):
#            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#            kmeans.fit(self.IMAGE)
#            wcss.append(kmeans.inertia_)
#        self.WCSS = wcss
#        plt.plot(range(1, 10), wcss)
#        plt.title('The Elbow Method')
#        plt.xlabel('Number of clusters')
#        plt.ylabel('WCSS')
#        plt.show()
        
#    def findK(self):
#        m = (self.WCSS[0]-self.WCSS[8])/(1-9)
#        c = self.WCSS[0]-m*1
#        A=m
#        B=-1
#        maxD = 0
#        K = 0
#        for x1,y1 in enumerate(self.WCSS,1):
#            d = abs(A*x1+B*y1+c)/math.sqrt(A**2+B**2)
#            if d > maxD:
#                maxD = d
#                self.CLUSTERS=x1+1
#        print('found :'+str(self.CLUSTERS)+' group')
#        return self.CLUSTERS


PATH = 'segmentation_WBC-master/Fulltest2/set4/'
        
ip = PATH+'compare/'
orig = PATH+'testset/'

all_files = next(os.walk(ip))[2]
orig_files = next(os.walk(orig))[2]

for idx in range (len(all_files)):
    if all_files[idx].find('.png') > 0:
        dc = DominantColors(ip+all_files[idx])
        dc.elbow()
        dc.dominantColors()
        
        neucleus = np.reshape(dc.LABELS, (-1, dc.ORIG.shape[1]))
        cytoplasm = neucleus.copy()
        cell = neucleus.copy()

        label = dc.KMEAN.predict([[125,23,212]])[0]
        for i in range(0,neucleus.shape[0]):
            for j in range(0,neucleus.shape[1]):
                if neucleus[i][j] == dc.pLABEL[0] :
                    neucleus[i][j] = 255
                else:
                    neucleus[i][j] = 0
                if cytoplasm[i][j] == label:
                    cytoplasm[i][j] = 255
                else:
                    cytoplasm[i][j] = 0

        opened = cytoplasm+neucleus
        for i in range(0,opened.shape[0]):
            for j in range(0,opened.shape[1]):
                if opened[i][j] <200:
                    opened[i][j] =0
        opened = opened.astype(bool)
        opened = closing(opened, square(2))
        
        opened = remove_small_objects(opened, 200)
        opened = remove_small_holes(opened,1600)
        
        img = cv2.imread(orig+str(orig_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                if opened[i][j] ==0 :
                    img[i][j] = 0
                    
        imsave(PATH+all_files[idx]+'.png',img)



































