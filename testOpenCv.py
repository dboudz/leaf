3# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


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

#img = cv2.imread('/Users/dboudeau/depot/leaf/maple-leaf-888807_960_720.jpg')
#img = cv2.imread('/Users/dboudeau/depot/leaf/opencv-python-foreground-extraction-tutorial.jpg')
#img = cv2.imread('/Users/dboudeau/depot/leaf/lind-ungt-lov.jpg')
img = cv2.imread('/Users/dboudeau/depot/leaf/lind-ungt-lov.jpg')
img0=img


plt.imshow(img0)


mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# ex
#rect = (161,79,150,150)
# maple-leaf-888807_960_720.jpg mine startx starty endx end y
#rect = (0,0,800,900)
# test 2 lind-ungt-lov.jpg
rect = (0,0,3000,4000)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

#plt.imshow(img)
#plt.colorbar()
#plt.show()

show_images([img0,img], cols = 2)
