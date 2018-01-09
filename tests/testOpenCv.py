# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


FOLDER_TRAINING_SET='/Users/dboudeau/depot/leaf/'
FARGESIA=os.listdir(FOLDER_TRAINING_SET+'FARGESIA/resized/') 


for image_name in FARGESIA:

    #img = cv2.imread(FOLDER_TRAINING_SET+'FARGESIA/resized/'+image_name)
    img=cv2.imread(FOLDER_TRAINING_SET+'FARGESIA/resized/'+'101.jpg')
    print("Processing "+str(image_name))
    img0=img
    
    
    #plt.imshow(img0)
    
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    list_img=[]
    for i in range(50,400,50):
        print(i)
        #X,Y <-> 
        rect = (0,0,i,i)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        imgtr = img0*mask2[:,:,np.newaxis]
        #cv2.rectangle(imgtr,(0,0),(i,i),(0,255,0),3)
        list_img.append(imgtr)

    
    #show_images([img0,img1,img2,img3], cols = 3)
    cmpt=0
    total=None
    for im in list_img:
        cv2.imshow("list_img["+str(cmpt)+"]",im)
        if(total is None):
            total=im
        else:
            total=total+im
        cmpt=cmpt+1
    cv2.imshow("init",img0)
    plt.imshow(img0)
    
    minus0=list_img[1]-list_img[0]
    minus1=list_img[2]-minus0
    minus2=list_img[3]-minus1
    minus3=list_img[4]-minus2
    minus4=list_img[5]-minus3
    minus5=list_img[6]-minus4               
    cv2.imshow("minus5",minus5)
    total=total+img0
    cv2.imshow("TOTAL",total)
    
    
    gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    
    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]
    
    #plt.imshow(minus5)
    
    break;
