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
import os
from sklearn.cluster import KMeans
import math

PATH = 'segmentation_WBC-master/'
all_files = next(os.walk(PATH+'Dataset 1'))[2]
img = color.rgb2hsv(imread(PATH+'Dataset 1/'+'003.bmp'))

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    ORIG = None
    WCSS = None
    pLABEL = None
    
    def __init__(self, image):
        self.IMAGE = image
        
    def dominantColors(self):
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS, init = 'k-means++', random_state = 0)
        kmeans.fit(self.IMAGE)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #find label of purple color
        self.pLABEL = kmeans.predict([[128,0,128]])
        #self.pLABEL = kmeans.predict([[150,255,127]])
        
        #returning after converting to integer from float
        return self.COLORS
    
    def show(self):
        imshow(self.ORIG)
        
    def elbow(self):
        #read image
        img = cv2.imread(self.IMAGE)
        
        #convert to rgb from bgr
        #img = color.rgb2hsv(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.ORIG = img.copy()
            
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #Elbow method
        wcss = []
        for i in range(1, 10):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(self.IMAGE)
            wcss.append(kmeans.inertia_)
        self.WCSS = wcss
#        plt.plot(range(1, 10), wcss)
#        plt.title('The Elbow Method')
#        plt.xlabel('Number of clusters')
#        plt.ylabel('WCSS')
#        plt.show()
        
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
    imsave(PATH+'/Kmean/'+'out_'+file+'.png',new)
        
def contour_sep():
    img = color.rgb2grey(imread('segmentation_WBC-master/Dataset 1/'+file))
    #invert black->white , white->black
    invert_img = 1-img
    
    #find contour ( edge )
    contours = measure.find_contours(invert_img, 0.7)
    '''
    # save contour plot
    fig, ax = plt.subplots()
    ax.imshow(invert_img, cmap=plt.cm.gray)
    
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('contour_'+file+'.png',bbox_inches='tight',pad_inches=0)
    '''
    for n, contour in enumerate(contours):
        pr = np.array([p[0] for p in contour])
        pc = np.array([p[1] for p in contour])
        row = list(map(np.uint8,contour[:,0]))
        col = list(map(np.uint8,contour[:,1]))
        rr, cc = polygon(pr,pc)
        #print(pr,',',pc)
        invert_img[rr,cc] = 255
    
    #convert gray -> binary
    thresh = threshold_mean(invert_img)
    binary = invert_img > thresh
    binary = binary*255
    imsave(PATH+'/segmentation/'+'out_'+file+'.png',binary)
    
    #label object in image
    label_img = label(binary)
    props = regionprops(label_img)


for file in all_files:
    if file.find('.bmp') > 0:
        #contour_sep()
        K_sep()