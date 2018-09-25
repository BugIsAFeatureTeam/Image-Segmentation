# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:40:26 2018

@author: 58011256
"""

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

PATH = 'segmentation_WBC-master/'
all_files = next(os.walk(PATH+'Dataset 1'))[2]
img = color.rgb2hsv(imread(PATH+'Dataset 1/'+'003.bmp'))

for file in all_files:
    if file.find('.bmp') > 0:
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
        